import json
import logging
import datetime
import time
import random
import uuid
from typing import Optional, List

from flask import current_app
from sqlalchemy import func

from core.index.index import IndexBuilder
from core.model_providers.error import LLMBadRequestError, ProviderTokenNotInitError
from core.model_providers.model_factory import ModelFactory
from extensions.ext_redis import redis_client
from flask_login import current_user

from events.dataset_event import dataset_was_deleted
from events.document_event import document_was_deleted
from extensions.ext_database import db
from libs import helper
from models.account import Account
from models.dataset import Dataset, Document, DatasetQuery, DatasetProcessRule, AppDatasetJoin, DocumentSegment, \
    DatasetCollectionBinding
from models.model import UploadFile
from models.source import DataSourceBinding
from services.errors.account import NoPermissionError
from services.errors.dataset import DatasetNameDuplicateError
from services.errors.document import DocumentIndexingError
from services.errors.file import FileNotExistsError
from services.vector_service import VectorService
from tasks.clean_notion_document_task import clean_notion_document_task
from tasks.deal_dataset_vector_index_task import deal_dataset_vector_index_task
from tasks.document_indexing_task import document_indexing_task
from tasks.document_indexing_update_task import document_indexing_update_task
from tasks.create_segment_to_index_task import create_segment_to_index_task
from tasks.update_segment_index_task import update_segment_index_task
from tasks.recover_document_indexing_task import recover_document_indexing_task
from tasks.update_segment_keyword_index_task import update_segment_keyword_index_task
from tasks.delete_segment_from_index_task import delete_segment_from_index_task


class DatasetService:

    @staticmethod
    def get_datasets(page, per_page, provider="vendor", tenant_id=None, user=None):
        if user:
            permission_filter = db.or_(Dataset.created_by == user.id,
                                       Dataset.permission == 'all_team_members')
        else:
            permission_filter = Dataset.permission == 'all_team_members'
        datasets = Dataset.query.filter(
            db.and_(Dataset.provider == provider, Dataset.tenant_id == tenant_id, permission_filter)) \
            .order_by(Dataset.created_at.desc()) \
            .paginate(
            page=page,
            per_page=per_page,
            max_per_page=100,
            error_out=False
        )

        return datasets.items, datasets.total

    @staticmethod
    def get_process_rules(dataset_id):
        # get the latest process rule
        dataset_process_rule = db.session.query(DatasetProcessRule). \
            filter(DatasetProcessRule.dataset_id == dataset_id). \
            order_by(DatasetProcessRule.created_at.desc()). \
            limit(1). \
            one_or_none()
        if dataset_process_rule:
            mode = dataset_process_rule.mode
            rules = dataset_process_rule.rules_dict
        else:
            mode = DocumentService.DEFAULT_RULES['mode']
            rules = DocumentService.DEFAULT_RULES['rules']
        return {
            'mode': mode,
            'rules': rules
        }

    @staticmethod
    def get_datasets_by_ids(ids, tenant_id):
        datasets = Dataset.query.filter(Dataset.id.in_(ids),
                                        Dataset.tenant_id == tenant_id).paginate(
            page=1, per_page=len(ids), max_per_page=len(ids), error_out=False)
        return datasets.items, datasets.total

    @staticmethod
    def create_empty_dataset(tenant_id: str, name: str, indexing_technique: Optional[str], account: Account):
        # check if dataset name already exists
        if Dataset.query.filter_by(name=name, tenant_id=tenant_id).first():
            raise DatasetNameDuplicateError(
                f'Dataset with name {name} already exists.')
        embedding_model = None
        if indexing_technique == 'high_quality':
            embedding_model = ModelFactory.get_embedding_model(
                tenant_id=current_user.current_tenant_id
            )
        dataset = Dataset(name=name, indexing_technique=indexing_technique)
        # dataset = Dataset(name=name, provider=provider, config=config)
        dataset.created_by = account.id
        dataset.updated_by = account.id
        dataset.tenant_id = tenant_id
        dataset.embedding_model_provider = embedding_model.model_provider.provider_name if embedding_model else None
        dataset.embedding_model = embedding_model.name if embedding_model else None
        db.session.add(dataset)
        db.session.commit()
        return dataset

    @staticmethod
    def get_dataset(dataset_id):
        dataset = Dataset.query.filter_by(
            id=dataset_id
        ).first()
        if dataset is None:
            return None
        else:
            return dataset

    @staticmethod
    def check_dataset_model_setting(dataset):
        if dataset.indexing_technique == 'high_quality':
            try:
                ModelFactory.get_embedding_model(
                    tenant_id=dataset.tenant_id,
                    model_provider_name=dataset.embedding_model_provider,
                    model_name=dataset.embedding_model
                )
            except LLMBadRequestError:
                raise ValueError(
                    f"No Embedding Model available. Please configure a valid provider "
                    f"in the Settings -> Model Provider.")
            except ProviderTokenNotInitError as ex:
                raise ValueError(f"The dataset in unavailable, due to: "
                                 f"{ex.description}")

    @staticmethod
    def update_dataset(dataset_id, data, user):
        filtered_data = {k: v for k, v in data.items() if v is not None or k == 'description'}
        dataset = DatasetService.get_dataset(dataset_id)
        DatasetService.check_dataset_permission(dataset, user)
        action = None
        if dataset.indexing_technique != data['indexing_technique']:
            # if update indexing_technique
            if data['indexing_technique'] == 'economy':
                action = 'remove'
                filtered_data['embedding_model'] = None
                filtered_data['embedding_model_provider'] = None
                filtered_data['collection_binding_id'] = None
            elif data['indexing_technique'] == 'high_quality':
                action = 'add'
                # get embedding model setting
                try:
                    embedding_model = ModelFactory.get_embedding_model(
                        tenant_id=current_user.current_tenant_id
                    )
                    filtered_data['embedding_model'] = embedding_model.name
                    filtered_data['embedding_model_provider'] = embedding_model.model_provider.provider_name
                    dataset_collection_binding = DatasetCollectionBindingService.get_dataset_collection_binding(
                        embedding_model.model_provider.provider_name,
                        embedding_model.name
                    )
                    filtered_data['collection_binding_id'] = dataset_collection_binding.id
                except LLMBadRequestError:
                    raise ValueError(
                        f"No Embedding Model available. Please configure a valid provider "
                        f"in the Settings -> Model Provider.")
                except ProviderTokenNotInitError as ex:
                    raise ValueError(ex.description)

        filtered_data['updated_by'] = user.id
        filtered_data['updated_at'] = datetime.datetime.now()

        dataset.query.filter_by(id=dataset_id).update(filtered_data)

        db.session.commit()
        if action:
            deal_dataset_vector_index_task.delay(dataset_id, action)
        return dataset

    @staticmethod
    def delete_dataset(dataset_id, user):
        # todo: cannot delete dataset if it is being processed

        dataset = DatasetService.get_dataset(dataset_id)

        if dataset is None:
            return False

        DatasetService.check_dataset_permission(dataset, user)

        dataset_was_deleted.send(dataset)

        db.session.delete(dataset)
        db.session.commit()
        return True

    @staticmethod
    def check_dataset_permission(dataset, user):
        if dataset.tenant_id != user.current_tenant_id:
            logging.debug(
                f'User {user.id} does not have permission to access dataset {dataset.id}')
            raise NoPermissionError(
                'You do not have permission to access this dataset.')
        if dataset.permission == 'only_me' and dataset.created_by != user.id:
            logging.debug(
                f'User {user.id} does not have permission to access dataset {dataset.id}')
            raise NoPermissionError(
                'You do not have permission to access this dataset.')

    @staticmethod
    def get_dataset_queries(dataset_id: str, page: int, per_page: int):
        dataset_queries = DatasetQuery.query.filter_by(dataset_id=dataset_id) \
            .order_by(db.desc(DatasetQuery.created_at)) \
            .paginate(
            page=page, per_page=per_page, max_per_page=100, error_out=False
        )
        return dataset_queries.items, dataset_queries.total

    @staticmethod
    def get_related_apps(dataset_id: str):
        return AppDatasetJoin.query.filter(AppDatasetJoin.dataset_id == dataset_id) \
            .order_by(db.desc(AppDatasetJoin.created_at)).all()


class DocumentService:
    DEFAULT_RULES = {
        'mode': 'custom',
        'rules': {
            'pre_processing_rules': [
                {'id': 'remove_extra_spaces', 'enabled': True},
                {'id': 'remove_urls_emails', 'enabled': False}
            ],
            'segmentation': {
                'delimiter': '\n',
                'max_tokens': 500
            }
        }
    }

    DOCUMENT_METADATA_SCHEMA = {
        "book": {
            "title": str,
            "language": str,
            "author": str,
            "publisher": str,
            "publication_date": str,
            "isbn": str,
            "category": str,
        },
        "web_page": {
            "title": str,
            "url": str,
            "language": str,
            "publish_date": str,
            "author/publisher": str,
            "topic/keywords": str,
            "description": str,
        },
        "paper": {
            "title": str,
            "language": str,
            "author": str,
            "publish_date": str,
            "journal/conference_name": str,
            "volume/issue/page_numbers": str,
            "doi": str,
            "topic/keywords": str,
            "abstract": str,
        },
        "social_media_post": {
            "platform": str,
            "author/username": str,
            "publish_date": str,
            "post_url": str,
            "topic/tags": str,
        },
        "wikipedia_entry": {
            "title": str,
            "language": str,
            "web_page_url": str,
            "last_edit_date": str,
            "editor/contributor": str,
            "summary/introduction": str,
        },
        "personal_document": {
            "title": str,
            "author": str,
            "creation_date": str,
            "last_modified_date": str,
            "document_type": str,
            "tags/category": str,
        },
        "business_document": {
            "title": str,
            "author": str,
            "creation_date": str,
            "last_modified_date": str,
            "document_type": str,
            "department/team": str,
        },
        "im_chat_log": {
            "chat_platform": str,
            "chat_participants/group_name": str,
            "start_date": str,
            "end_date": str,
            "summary": str,
        },
        "synced_from_notion": {
            "title": str,
            "language": str,
            "author/creator": str,
            "creation_date": str,
            "last_modified_date": str,
            "notion_page_link": str,
            "category/tags": str,
            "description": str,
        },
        "synced_from_github": {
            "repository_name": str,
            "repository_description": str,
            "repository_owner/organization": str,
            "code_filename": str,
            "code_file_path": str,
            "programming_language": str,
            "github_link": str,
            "open_source_license": str,
            "commit_date": str,
            "commit_author": str,
        },
        "others": dict
    }

    @staticmethod
    def get_document(dataset_id: str, document_id: str) -> Optional[Document]:
        document = db.session.query(Document).filter(
            Document.id == document_id,
            Document.dataset_id == dataset_id
        ).first()

        return document

    @staticmethod
    def get_document_by_id(document_id: str) -> Optional[Document]:
        document = db.session.query(Document).filter(
            Document.id == document_id
        ).first()

        return document

    @staticmethod
    def get_document_by_dataset_id(dataset_id: str) -> List[Document]:
        documents = db.session.query(Document).filter(
            Document.dataset_id == dataset_id,
            Document.enabled == True
        ).all()

        return documents

    @staticmethod
    def get_batch_documents(dataset_id: str, batch: str) -> List[Document]:
        documents = db.session.query(Document).filter(
            Document.batch == batch,
            Document.dataset_id == dataset_id,
            Document.tenant_id == current_user.current_tenant_id
        ).all()

        return documents

    @staticmethod
    def get_document_file_detail(file_id: str):
        file_detail = db.session.query(UploadFile). \
            filter(UploadFile.id == file_id). \
            one_or_none()
        return file_detail

    @staticmethod
    def check_archived(document):
        if document.archived:
            return True
        else:
            return False

    @staticmethod
    def delete_document(document):
        if document.indexing_status in ["parsing", "cleaning", "splitting", "indexing"]:
            raise DocumentIndexingError()

        # trigger document_was_deleted signal
        document_was_deleted.send(document.id, dataset_id=document.dataset_id)

        db.session.delete(document)
        db.session.commit()

    @staticmethod
    def pause_document(document):
        if document.indexing_status not in ["waiting", "parsing", "cleaning", "splitting", "indexing"]:
            raise DocumentIndexingError()
        # update document to be paused
        document.is_paused = True
        document.paused_by = current_user.id
        document.paused_at = datetime.datetime.utcnow()

        db.session.add(document)
        db.session.commit()
        # set document paused flag
        indexing_cache_key = 'document_{}_is_paused'.format(document.id)
        redis_client.setnx(indexing_cache_key, "True")

    @staticmethod
    def recover_document(document):
        if not document.is_paused:
            raise DocumentIndexingError()
        # update document to be recover
        document.is_paused = False
        document.paused_by = None
        document.paused_at = None

        db.session.add(document)
        db.session.commit()
        # delete paused flag
        indexing_cache_key = 'document_{}_is_paused'.format(document.id)
        redis_client.delete(indexing_cache_key)
        # trigger async task
        recover_document_indexing_task.delay(document.dataset_id, document.id)

    @staticmethod
    def get_documents_position(dataset_id):
        document = Document.query.filter_by(dataset_id=dataset_id).order_by(Document.position.desc()).first()
        if document:
            return document.position + 1
        else:
            return 1

    @staticmethod
    def save_document_with_dataset_id(dataset: Dataset, document_data: dict,
                                      account: Account, dataset_process_rule: Optional[DatasetProcessRule] = None,
                                      created_from: str = 'web'):

        # check document limit
        if current_app.config['EDITION'] == 'CLOUD':
            if 'original_document_id' not in document_data or not document_data['original_document_id']:
                count = 0
                if document_data["data_source"]["type"] == "upload_file":
                    upload_file_list = document_data["data_source"]["info_list"]['file_info_list']['file_ids']
                    count = len(upload_file_list)
                elif document_data["data_source"]["type"] == "notion_import":
                    notion_info_list = document_data["data_source"]['info_list']['notion_info_list']
                    for notion_info in notion_info_list:
                        count = count + len(notion_info['pages'])
                documents_count = DocumentService.get_tenant_documents_count()
                total_count = documents_count + count
                tenant_document_count = int(current_app.config['TENANT_DOCUMENT_COUNT'])
                if total_count > tenant_document_count:
                    raise ValueError(f"over document limit {tenant_document_count}.")
        # if dataset is empty, update dataset data_source_type
        if not dataset.data_source_type:
            dataset.data_source_type = document_data["data_source"]["type"]

        if not dataset.indexing_technique:
            if 'indexing_technique' not in document_data \
                    or document_data['indexing_technique'] not in Dataset.INDEXING_TECHNIQUE_LIST:
                raise ValueError("Indexing technique is required")

            dataset.indexing_technique = document_data["indexing_technique"]
            if document_data["indexing_technique"] == 'high_quality':
                embedding_model = ModelFactory.get_embedding_model(
                    tenant_id=dataset.tenant_id
                )
                dataset.embedding_model = embedding_model.name
                dataset.embedding_model_provider = embedding_model.model_provider.provider_name
                dataset_collection_binding = DatasetCollectionBindingService.get_dataset_collection_binding(
                    embedding_model.model_provider.provider_name,
                    embedding_model.name
                )
                dataset.collection_binding_id = dataset_collection_binding.id

        documents = []
        batch = time.strftime('%Y%m%d%H%M%S') + str(random.randint(100000, 999999))
        if 'original_document_id' in document_data and document_data["original_document_id"]:
            document = DocumentService.update_document_with_dataset_id(dataset, document_data, account)
            documents.append(document)
        else:
            # save process rule
            if not dataset_process_rule:
                process_rule = document_data["process_rule"]
                if process_rule["mode"] == "custom":
                    dataset_process_rule = DatasetProcessRule(
                        dataset_id=dataset.id,
                        mode=process_rule["mode"],
                        rules=json.dumps(process_rule["rules"]),
                        created_by=account.id
                    )
                elif process_rule["mode"] == "automatic":
                    dataset_process_rule = DatasetProcessRule(
                        dataset_id=dataset.id,
                        mode=process_rule["mode"],
                        rules=json.dumps(DatasetProcessRule.AUTOMATIC_RULES),
                        created_by=account.id
                    )
                db.session.add(dataset_process_rule)
                db.session.commit()
            position = DocumentService.get_documents_position(dataset.id)
            document_ids = []
            if document_data["data_source"]["type"] == "upload_file":
                upload_file_list = document_data["data_source"]["info_list"]['file_info_list']['file_ids']
                for file_id in upload_file_list:
                    file = db.session.query(UploadFile).filter(
                        UploadFile.tenant_id == dataset.tenant_id,
                        UploadFile.id == file_id
                    ).first()

                    # raise error if file not found
                    if not file:
                        raise FileNotExistsError()

                    file_name = file.name
                    data_source_info = {
                        "upload_file_id": file_id,
                    }
                    document = DocumentService.build_document(dataset, dataset_process_rule.id,
                                                              document_data["data_source"]["type"],
                                                              document_data["doc_form"],
                                                              document_data["doc_language"],
                                                              data_source_info, created_from, position,
                                                              account, file_name, batch)
                    db.session.add(document)
                    db.session.flush()
                    document_ids.append(document.id)
                    documents.append(document)
                    position += 1
            elif document_data["data_source"]["type"] == "notion_import":
                notion_info_list = document_data["data_source"]['info_list']['notion_info_list']
                exist_page_ids = []
                exist_document = dict()
                documents = Document.query.filter_by(
                    dataset_id=dataset.id,
                    tenant_id=current_user.current_tenant_id,
                    data_source_type='notion_import',
                    enabled=True
                ).all()
                if documents:
                    for document in documents:
                        data_source_info = json.loads(document.data_source_info)
                        exist_page_ids.append(data_source_info['notion_page_id'])
                        exist_document[data_source_info['notion_page_id']] = document.id
                for notion_info in notion_info_list:
                    workspace_id = notion_info['workspace_id']
                    data_source_binding = DataSourceBinding.query.filter(
                        db.and_(
                            DataSourceBinding.tenant_id == current_user.current_tenant_id,
                            DataSourceBinding.provider == 'notion',
                            DataSourceBinding.disabled == False,
                            DataSourceBinding.source_info['workspace_id'] == f'"{workspace_id}"'
                        )
                    ).first()
                    if not data_source_binding:
                        raise ValueError('Data source binding not found.')
                    for page in notion_info['pages']:
                        if page['page_id'] not in exist_page_ids:
                            data_source_info = {
                                "notion_workspace_id": workspace_id,
                                "notion_page_id": page['page_id'],
                                "notion_page_icon": page['page_icon'],
                                "type": page['type']
                            }
                            document = DocumentService.build_document(dataset, dataset_process_rule.id,
                                                                      document_data["data_source"]["type"],
                                                                      document_data["doc_form"],
                                                                      document_data["doc_language"],
                                                                      data_source_info, created_from, position,
                                                                      account, page['page_name'], batch)
                            db.session.add(document)
                            db.session.flush()
                            document_ids.append(document.id)
                            documents.append(document)
                            position += 1
                        else:
                            exist_document.pop(page['page_id'])
                # delete not selected documents
                if len(exist_document) > 0:
                    clean_notion_document_task.delay(list(exist_document.values()), dataset.id)
            db.session.commit()

            # trigger async task
            document_indexing_task.delay(dataset.id, document_ids)

        return documents, batch

    @staticmethod
    def build_document(dataset: Dataset, process_rule_id: str, data_source_type: str, document_form: str,
                       document_language: str, data_source_info: dict, created_from: str, position: int,
                       account: Account,
                       name: str, batch: str):
        document = Document(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            position=position,
            data_source_type=data_source_type,
            data_source_info=json.dumps(data_source_info),
            dataset_process_rule_id=process_rule_id,
            batch=batch,
            name=name,
            created_from=created_from,
            created_by=account.id,
            doc_form=document_form,
            doc_language=document_language
        )
        return document

    @staticmethod
    def get_tenant_documents_count():
        documents_count = Document.query.filter(Document.completed_at.isnot(None),
                                                Document.enabled == True,
                                                Document.archived == False,
                                                Document.tenant_id == current_user.current_tenant_id).count()
        return documents_count

    @staticmethod
    def update_document_with_dataset_id(dataset: Dataset, document_data: dict,
                                        account: Account, dataset_process_rule: Optional[DatasetProcessRule] = None,
                                        created_from: str = 'web'):
        DatasetService.check_dataset_model_setting(dataset)
        document = DocumentService.get_document(dataset.id, document_data["original_document_id"])
        if document.display_status != 'available':
            raise ValueError("Document is not available")
        # save process rule
        if 'process_rule' in document_data and document_data['process_rule']:
            process_rule = document_data["process_rule"]
            if process_rule["mode"] == "custom":
                dataset_process_rule = DatasetProcessRule(
                    dataset_id=dataset.id,
                    mode=process_rule["mode"],
                    rules=json.dumps(process_rule["rules"]),
                    created_by=account.id
                )
            elif process_rule["mode"] == "automatic":
                dataset_process_rule = DatasetProcessRule(
                    dataset_id=dataset.id,
                    mode=process_rule["mode"],
                    rules=json.dumps(DatasetProcessRule.AUTOMATIC_RULES),
                    created_by=account.id
                )
            db.session.add(dataset_process_rule)
            db.session.commit()
            document.dataset_process_rule_id = dataset_process_rule.id
        # update document data source
        if 'data_source' in document_data and document_data['data_source']:
            file_name = ''
            data_source_info = {}
            if document_data["data_source"]["type"] == "upload_file":
                upload_file_list = document_data["data_source"]["info_list"]['file_info_list']['file_ids']
                for file_id in upload_file_list:
                    file = db.session.query(UploadFile).filter(
                        UploadFile.tenant_id == dataset.tenant_id,
                        UploadFile.id == file_id
                    ).first()

                    # raise error if file not found
                    if not file:
                        raise FileNotExistsError()

                    file_name = file.name
                    data_source_info = {
                        "upload_file_id": file_id,
                    }
            elif document_data["data_source"]["type"] == "notion_import":
                notion_info_list = document_data["data_source"]['info_list']['notion_info_list']
                for notion_info in notion_info_list:
                    workspace_id = notion_info['workspace_id']
                    data_source_binding = DataSourceBinding.query.filter(
                        db.and_(
                            DataSourceBinding.tenant_id == current_user.current_tenant_id,
                            DataSourceBinding.provider == 'notion',
                            DataSourceBinding.disabled == False,
                            DataSourceBinding.source_info['workspace_id'] == f'"{workspace_id}"'
                        )
                    ).first()
                    if not data_source_binding:
                        raise ValueError('Data source binding not found.')
                    for page in notion_info['pages']:
                        data_source_info = {
                            "notion_workspace_id": workspace_id,
                            "notion_page_id": page['page_id'],
                            "notion_page_icon": page['page_icon'],
                            "type": page['type']
                        }
            document.data_source_type = document_data["data_source"]["type"]
            document.data_source_info = json.dumps(data_source_info)
            document.name = file_name
        # update document to be waiting
        document.indexing_status = 'waiting'
        document.completed_at = None
        document.processing_started_at = None
        document.parsing_completed_at = None
        document.cleaning_completed_at = None
        document.splitting_completed_at = None
        document.updated_at = datetime.datetime.utcnow()
        document.created_from = created_from
        document.doc_form = document_data['doc_form']
        db.session.add(document)
        db.session.commit()
        # update document segment
        update_params = {
            DocumentSegment.status: 're_segment'
        }
        DocumentSegment.query.filter_by(document_id=document.id).update(update_params)
        db.session.commit()
        # trigger async task
        document_indexing_update_task.delay(document.dataset_id, document.id)

        return document

    @staticmethod
    def save_document_without_dataset_id(tenant_id: str, document_data: dict, account: Account):
        count = 0
        if document_data["data_source"]["type"] == "upload_file":
            upload_file_list = document_data["data_source"]["info_list"]['file_info_list']['file_ids']
            count = len(upload_file_list)
        elif document_data["data_source"]["type"] == "notion_import":
            notion_info_list = document_data["data_source"]['info_list']['notion_info_list']
            for notion_info in notion_info_list:
                count = count + len(notion_info['pages'])
        # check document limit
        if current_app.config['EDITION'] == 'CLOUD':
            documents_count = DocumentService.get_tenant_documents_count()
            total_count = documents_count + count
            tenant_document_count = int(current_app.config['TENANT_DOCUMENT_COUNT'])
            if total_count > tenant_document_count:
                raise ValueError(f"All your documents have overed limit {tenant_document_count}.")
        embedding_model = None
        dataset_collection_binding_id = None
        if document_data['indexing_technique'] == 'high_quality':
            embedding_model = ModelFactory.get_embedding_model(
                tenant_id=tenant_id
            )
            dataset_collection_binding = DatasetCollectionBindingService.get_dataset_collection_binding(
                embedding_model.model_provider.provider_name,
                embedding_model.name
            )
            dataset_collection_binding_id = dataset_collection_binding.id
        # save dataset
        dataset = Dataset(
            tenant_id=tenant_id,
            name='',
            data_source_type=document_data["data_source"]["type"],
            indexing_technique=document_data["indexing_technique"],
            created_by=account.id,
            embedding_model=embedding_model.name if embedding_model else None,
            embedding_model_provider=embedding_model.model_provider.provider_name if embedding_model else None,
            collection_binding_id=dataset_collection_binding_id
        )

        db.session.add(dataset)
        db.session.flush()

        documents, batch = DocumentService.save_document_with_dataset_id(dataset, document_data, account)

        cut_length = 18
        cut_name = documents[0].name[:cut_length]
        dataset.name = cut_name + '...'
        dataset.description = 'useful for when you want to answer queries about the ' + documents[0].name
        db.session.commit()

        return dataset, documents, batch

    @classmethod
    def document_create_args_validate(cls, args: dict):
        if 'original_document_id' not in args or not args['original_document_id']:
            DocumentService.data_source_args_validate(args)
            DocumentService.process_rule_args_validate(args)
        else:
            if ('data_source' not in args and not args['data_source']) \
                    and ('process_rule' not in args and not args['process_rule']):
                raise ValueError("Data source or Process rule is required")
            else:
                if 'data_source' in args and args['data_source']:
                    DocumentService.data_source_args_validate(args)
                if 'process_rule' in args and args['process_rule']:
                    DocumentService.process_rule_args_validate(args)

    @classmethod
    def data_source_args_validate(cls, args: dict):
        if 'data_source' not in args or not args['data_source']:
            raise ValueError("Data source is required")

        if not isinstance(args['data_source'], dict):
            raise ValueError("Data source is invalid")

        if 'type' not in args['data_source'] or not args['data_source']['type']:
            raise ValueError("Data source type is required")

        if args['data_source']['type'] not in Document.DATA_SOURCES:
            raise ValueError("Data source type is invalid")

        if 'info_list' not in args['data_source'] or not args['data_source']['info_list']:
            raise ValueError("Data source info is required")

        if args['data_source']['type'] == 'upload_file':
            if 'file_info_list' not in args['data_source']['info_list'] or not args['data_source']['info_list'][
                'file_info_list']:
                raise ValueError("File source info is required")
        if args['data_source']['type'] == 'notion_import':
            if 'notion_info_list' not in args['data_source']['info_list'] or not args['data_source']['info_list'][
                'notion_info_list']:
                raise ValueError("Notion source info is required")

    @classmethod
    def process_rule_args_validate(cls, args: dict):
        if 'process_rule' not in args or not args['process_rule']:
            raise ValueError("Process rule is required")

        if not isinstance(args['process_rule'], dict):
            raise ValueError("Process rule is invalid")

        if 'mode' not in args['process_rule'] or not args['process_rule']['mode']:
            raise ValueError("Process rule mode is required")

        if args['process_rule']['mode'] not in DatasetProcessRule.MODES:
            raise ValueError("Process rule mode is invalid")

        if args['process_rule']['mode'] == 'automatic':
            args['process_rule']['rules'] = {}
        else:
            if 'rules' not in args['process_rule'] or not args['process_rule']['rules']:
                raise ValueError("Process rule rules is required")

            if not isinstance(args['process_rule']['rules'], dict):
                raise ValueError("Process rule rules is invalid")

            if 'pre_processing_rules' not in args['process_rule']['rules'] \
                    or args['process_rule']['rules']['pre_processing_rules'] is None:
                raise ValueError("Process rule pre_processing_rules is required")

            if not isinstance(args['process_rule']['rules']['pre_processing_rules'], list):
                raise ValueError("Process rule pre_processing_rules is invalid")

            unique_pre_processing_rule_dicts = {}
            for pre_processing_rule in args['process_rule']['rules']['pre_processing_rules']:
                if 'id' not in pre_processing_rule or not pre_processing_rule['id']:
                    raise ValueError("Process rule pre_processing_rules id is required")

                if pre_processing_rule['id'] not in DatasetProcessRule.PRE_PROCESSING_RULES:
                    raise ValueError("Process rule pre_processing_rules id is invalid")

                if 'enabled' not in pre_processing_rule or pre_processing_rule['enabled'] is None:
                    raise ValueError("Process rule pre_processing_rules enabled is required")

                if not isinstance(pre_processing_rule['enabled'], bool):
                    raise ValueError("Process rule pre_processing_rules enabled is invalid")

                unique_pre_processing_rule_dicts[pre_processing_rule['id']] = pre_processing_rule

            args['process_rule']['rules']['pre_processing_rules'] = list(unique_pre_processing_rule_dicts.values())

            if 'segmentation' not in args['process_rule']['rules'] \
                    or args['process_rule']['rules']['segmentation'] is None:
                raise ValueError("Process rule segmentation is required")

            if not isinstance(args['process_rule']['rules']['segmentation'], dict):
                raise ValueError("Process rule segmentation is invalid")

            if 'separator' not in args['process_rule']['rules']['segmentation'] \
                    or not args['process_rule']['rules']['segmentation']['separator']:
                raise ValueError("Process rule segmentation separator is required")

            if not isinstance(args['process_rule']['rules']['segmentation']['separator'], str):
                raise ValueError("Process rule segmentation separator is invalid")

            if 'max_tokens' not in args['process_rule']['rules']['segmentation'] \
                    or not args['process_rule']['rules']['segmentation']['max_tokens']:
                raise ValueError("Process rule segmentation max_tokens is required")

            if not isinstance(args['process_rule']['rules']['segmentation']['max_tokens'], int):
                raise ValueError("Process rule segmentation max_tokens is invalid")

    @classmethod
    def estimate_args_validate(cls, args: dict):
        if 'info_list' not in args or not args['info_list']:
            raise ValueError("Data source info is required")

        if not isinstance(args['info_list'], dict):
            raise ValueError("Data info is invalid")

        if 'process_rule' not in args or not args['process_rule']:
            raise ValueError("Process rule is required")

        if not isinstance(args['process_rule'], dict):
            raise ValueError("Process rule is invalid")

        if 'mode' not in args['process_rule'] or not args['process_rule']['mode']:
            raise ValueError("Process rule mode is required")

        if args['process_rule']['mode'] not in DatasetProcessRule.MODES:
            raise ValueError("Process rule mode is invalid")

        if args['process_rule']['mode'] == 'automatic':
            args['process_rule']['rules'] = {}
        else:
            if 'rules' not in args['process_rule'] or not args['process_rule']['rules']:
                raise ValueError("Process rule rules is required")

            if not isinstance(args['process_rule']['rules'], dict):
                raise ValueError("Process rule rules is invalid")

            if 'pre_processing_rules' not in args['process_rule']['rules'] \
                    or args['process_rule']['rules']['pre_processing_rules'] is None:
                raise ValueError("Process rule pre_processing_rules is required")

            if not isinstance(args['process_rule']['rules']['pre_processing_rules'], list):
                raise ValueError("Process rule pre_processing_rules is invalid")

            unique_pre_processing_rule_dicts = {}
            for pre_processing_rule in args['process_rule']['rules']['pre_processing_rules']:
                if 'id' not in pre_processing_rule or not pre_processing_rule['id']:
                    raise ValueError("Process rule pre_processing_rules id is required")

                if pre_processing_rule['id'] not in DatasetProcessRule.PRE_PROCESSING_RULES:
                    raise ValueError("Process rule pre_processing_rules id is invalid")

                if 'enabled' not in pre_processing_rule or pre_processing_rule['enabled'] is None:
                    raise ValueError("Process rule pre_processing_rules enabled is required")

                if not isinstance(pre_processing_rule['enabled'], bool):
                    raise ValueError("Process rule pre_processing_rules enabled is invalid")

                unique_pre_processing_rule_dicts[pre_processing_rule['id']] = pre_processing_rule

            args['process_rule']['rules']['pre_processing_rules'] = list(unique_pre_processing_rule_dicts.values())

            if 'segmentation' not in args['process_rule']['rules'] \
                    or args['process_rule']['rules']['segmentation'] is None:
                raise ValueError("Process rule segmentation is required")

            if not isinstance(args['process_rule']['rules']['segmentation'], dict):
                raise ValueError("Process rule segmentation is invalid")

            if 'separator' not in args['process_rule']['rules']['segmentation'] \
                    or not args['process_rule']['rules']['segmentation']['separator']:
                raise ValueError("Process rule segmentation separator is required")

            if not isinstance(args['process_rule']['rules']['segmentation']['separator'], str):
                raise ValueError("Process rule segmentation separator is invalid")

            if 'max_tokens' not in args['process_rule']['rules']['segmentation'] \
                    or not args['process_rule']['rules']['segmentation']['max_tokens']:
                raise ValueError("Process rule segmentation max_tokens is required")

            if not isinstance(args['process_rule']['rules']['segmentation']['max_tokens'], int):
                raise ValueError("Process rule segmentation max_tokens is invalid")


class SegmentService:
    @classmethod
    def segment_create_args_validate(cls, args: dict, document: Document):
        if document.doc_form == 'qa_model':
            if 'answer' not in args or not args['answer']:
                raise ValueError("Answer is required")
            if not args['answer'].strip():
                raise ValueError("Answer is empty")
        if 'content' not in args or not args['content'] or not args['content'].strip():
            raise ValueError("Content is empty")

    @classmethod
    def create_segment(cls, args: dict, document: Document, dataset: Dataset):
        content = args['content']
        doc_id = str(uuid.uuid4())
        segment_hash = helper.generate_text_hash(content)
        tokens = 0
        if dataset.indexing_technique == 'high_quality':
            embedding_model = ModelFactory.get_embedding_model(
                tenant_id=dataset.tenant_id,
                model_provider_name=dataset.embedding_model_provider,
                model_name=dataset.embedding_model
            )
            # calc embedding use tokens
            tokens = embedding_model.get_num_tokens(content)
        max_position = db.session.query(func.max(DocumentSegment.position)).filter(
            DocumentSegment.document_id == document.id
        ).scalar()
        segment_document = DocumentSegment(
            tenant_id=current_user.current_tenant_id,
            dataset_id=document.dataset_id,
            document_id=document.id,
            index_node_id=doc_id,
            index_node_hash=segment_hash,
            position=max_position + 1 if max_position else 1,
            content=content,
            word_count=len(content),
            tokens=tokens,
            status='completed',
            indexing_at=datetime.datetime.utcnow(),
            completed_at=datetime.datetime.utcnow(),
            created_by=current_user.id
        )
        if document.doc_form == 'qa_model':
            segment_document.answer = args['answer']

        db.session.add(segment_document)
        db.session.commit()

        # save vector index
        try:
            VectorService.create_segment_vector(args['keywords'], segment_document, dataset)
        except Exception as e:
            logging.exception("create segment index failed")
            segment_document.enabled = False
            segment_document.disabled_at = datetime.datetime.utcnow()
            segment_document.status = 'error'
            segment_document.error = str(e)
            db.session.commit()
        segment = db.session.query(DocumentSegment).filter(DocumentSegment.id == segment_document.id).first()
        return segment

    @classmethod
    def update_segment(cls, args: dict, segment: DocumentSegment, document: Document, dataset: Dataset):
        indexing_cache_key = 'segment_{}_indexing'.format(segment.id)
        cache_result = redis_client.get(indexing_cache_key)
        if cache_result is not None:
            raise ValueError("Segment is indexing, please try again later")
        try:
            content = args['content']
            if segment.content == content:
                if document.doc_form == 'qa_model':
                    segment.answer = args['answer']
                if args['keywords']:
                    segment.keywords = args['keywords']
                db.session.add(segment)
                db.session.commit()
                # update segment index task
                if args['keywords']:
                    kw_index = IndexBuilder.get_index(dataset, 'economy')
                    # delete from keyword index
                    kw_index.delete_by_ids([segment.index_node_id])
                    # save keyword index
                    kw_index.update_segment_keywords_index(segment.index_node_id, segment.keywords)
            else:
                segment_hash = helper.generate_text_hash(content)
                tokens = 0
                if dataset.indexing_technique == 'high_quality':
                    embedding_model = ModelFactory.get_embedding_model(
                        tenant_id=dataset.tenant_id,
                        model_provider_name=dataset.embedding_model_provider,
                        model_name=dataset.embedding_model
                    )

                    # calc embedding use tokens
                    tokens = embedding_model.get_num_tokens(content)
                segment.content = content
                segment.index_node_hash = segment_hash
                segment.word_count = len(content)
                segment.tokens = tokens
                segment.status = 'completed'
                segment.indexing_at = datetime.datetime.utcnow()
                segment.completed_at = datetime.datetime.utcnow()
                segment.updated_by = current_user.id
                segment.updated_at = datetime.datetime.utcnow()
                if document.doc_form == 'qa_model':
                    segment.answer = args['answer']
                db.session.add(segment)
                db.session.commit()
                # update segment vector index
                VectorService.update_segment_vector(args['keywords'], segment, dataset)
        except Exception as e:
            logging.exception("update segment index failed")
            segment.enabled = False
            segment.disabled_at = datetime.datetime.utcnow()
            segment.status = 'error'
            segment.error = str(e)
            db.session.commit()
        segment = db.session.query(DocumentSegment).filter(DocumentSegment.id == segment.id).first()
        return segment

    @classmethod
    def delete_segment(cls, segment: DocumentSegment, document: Document, dataset: Dataset):
        indexing_cache_key = 'segment_{}_delete_indexing'.format(segment.id)
        cache_result = redis_client.get(indexing_cache_key)
        if cache_result is not None:
            raise ValueError("Segment is deleting.")

        # enabled segment need to delete index
        if segment.enabled:
            # send delete segment index task
            redis_client.setex(indexing_cache_key, 600, 1)
            delete_segment_from_index_task.delay(segment.id, segment.index_node_id, dataset.id, document.id)
        db.session.delete(segment)
        db.session.commit()


class DatasetCollectionBindingService:
    @classmethod
    def get_dataset_collection_binding(cls, provider_name: str, model_name: str) -> DatasetCollectionBinding:
        dataset_collection_binding = db.session.query(DatasetCollectionBinding). \
            filter(DatasetCollectionBinding.provider_name == provider_name,
                   DatasetCollectionBinding.model_name == model_name). \
            order_by(DatasetCollectionBinding.created_at). \
            first()

        if not dataset_collection_binding:
            dataset_collection_binding = DatasetCollectionBinding(
                provider_name=provider_name,
                model_name=model_name,
                collection_name="Vector_index_" + str(uuid.uuid4()).replace("-", "_") + '_Node'
            )
            db.session.add(dataset_collection_binding)
            db.session.flush()
        return dataset_collection_binding
