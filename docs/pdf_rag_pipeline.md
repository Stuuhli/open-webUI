PDF-RAG Pipeline (open-webui)
=============================

Diese Übersicht sammelt die zentralen Codepfade (1:1 kopiert) für das Einlesen von PDFs, das Chunken und Einbetten sowie das Retrieval. Kurze Hinweise vor jedem Block beschreiben, wie die Komponenten zusammenspielen.

---

backend/open_webui/retrieval/loaders/main.py — Loader._get_loader (PDF- und OCR-Pfade)

```python
    def _get_loader(self, filename: str, file_content_type: str, file_path: str):
        file_ext = filename.split(".")[-1].lower()

        if (
            self.engine == "external"
            and self.kwargs.get("EXTERNAL_DOCUMENT_LOADER_URL")
            and self.kwargs.get("EXTERNAL_DOCUMENT_LOADER_API_KEY")
        ):
            loader = ExternalDocumentLoader(
                file_path=file_path,
                url=self.kwargs.get("EXTERNAL_DOCUMENT_LOADER_URL"),
                api_key=self.kwargs.get("EXTERNAL_DOCUMENT_LOADER_API_KEY"),
                mime_type=file_content_type,
                user=self.user,
            )
        elif self.engine == "tika" and self.kwargs.get("TIKA_SERVER_URL"):
            if self._is_text_file(file_ext, file_content_type):
                loader = TextLoader(file_path, autodetect_encoding=True)
            else:
                loader = TikaLoader(
                    url=self.kwargs.get("TIKA_SERVER_URL"),
                    file_path=file_path,
                    extract_images=self.kwargs.get("PDF_EXTRACT_IMAGES"),
                )
        elif (
            self.engine == "datalab_marker"
            and self.kwargs.get("DATALAB_MARKER_API_KEY")
            and file_ext
            in [
                "pdf",
                "xls",
                "xlsx",
                "ods",
                "doc",
                "docx",
                "odt",
                "ppt",
                "pptx",
                "odp",
                "html",
                "epub",
                "png",
                "jpeg",
                "jpg",
                "webp",
                "gif",
                "tiff",
            ]
        ):
            api_base_url = self.kwargs.get("DATALAB_MARKER_API_BASE_URL", "")
            if not api_base_url or api_base_url.strip() == "":
                api_base_url = "https://www.datalab.to/api/v1/marker"  # https://github.com/open-webui/open-webui/pull/16867#issuecomment-3218424349

            loader = DatalabMarkerLoader(
                file_path=file_path,
                api_key=self.kwargs["DATALAB_MARKER_API_KEY"],
                api_base_url=api_base_url,
                additional_config=self.kwargs.get("DATALAB_MARKER_ADDITIONAL_CONFIG"),
                use_llm=self.kwargs.get("DATALAB_MARKER_USE_LLM", False),
                skip_cache=self.kwargs.get("DATALAB_MARKER_SKIP_CACHE", False),
                force_ocr=self.kwargs.get("DATALAB_MARKER_FORCE_OCR", False),
                paginate=self.kwargs.get("DATALAB_MARKER_PAGINATE", False),
                strip_existing_ocr=self.kwargs.get(
                    "DATALAB_MARKER_STRIP_EXISTING_OCR", False
                ),
                disable_image_extraction=self.kwargs.get(
                    "DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION", False
                ),
                format_lines=self.kwargs.get("DATALAB_MARKER_FORMAT_LINES", False),
                output_format=self.kwargs.get(
                    "DATALAB_MARKER_OUTPUT_FORMAT", "markdown"
                ),
            )
        elif self.engine == "docling" and self.kwargs.get("DOCLING_SERVER_URL"):
            if self._is_text_file(file_ext, file_content_type):
                loader = TextLoader(file_path, autodetect_encoding=True)
            else:
                # Build params for DoclingLoader
                params = self.kwargs.get("DOCLING_PARAMS", {})
                if not isinstance(params, dict):
                    try:
                        params = json.loads(params)
                    except json.JSONDecodeError:
                        log.error("Invalid DOCLING_PARAMS format, expected JSON object")
                        params = {}

                loader = DoclingLoader(
                    url=self.kwargs.get("DOCLING_SERVER_URL"),
                    file_path=file_path,
                    mime_type=file_content_type,
                    params=params,
                )
        elif (
            self.engine == "document_intelligence"
            and self.kwargs.get("DOCUMENT_INTELLIGENCE_ENDPOINT") != ""
            and (
                file_ext in ["pdf", "docx", "ppt", "pptx"]
                or file_content_type
                in [
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/vnd.ms-powerpoint",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ]
            )
        ):
            if self.kwargs.get("DOCUMENT_INTELLIGENCE_KEY") != "":
                loader = AzureAIDocumentIntelligenceLoader(
                    file_path=file_path,
                    api_endpoint=self.kwargs.get("DOCUMENT_INTELLIGENCE_ENDPOINT"),
                    api_key=self.kwargs.get("DOCUMENT_INTELLIGENCE_KEY"),
                )
            else:
                loader = AzureAIDocumentIntelligenceLoader(
                    file_path=file_path,
                    api_endpoint=self.kwargs.get("DOCUMENT_INTELLIGENCE_ENDPOINT"),
                    azure_credential=DefaultAzureCredential(),
                )
        elif self.engine == "mineru" and file_ext in [
            "pdf"
        ]:  # MinerU currently only supports PDF
            loader = MinerULoader(
                file_path=file_path,
                api_mode=self.kwargs.get("MINERU_API_MODE", "local"),
                api_url=self.kwargs.get("MINERU_API_URL", "http://localhost:8000"),
                api_key=self.kwargs.get("MINERU_API_KEY", ""),
                params=self.kwargs.get("MINERU_PARAMS", {}),
            )
        elif (
            self.engine == "mistral_ocr"
            and self.kwargs.get("MISTRAL_OCR_API_KEY") != ""
            and file_ext
            in ["pdf"]  # Mistral OCR currently only supports PDF and images
        ):
            loader = MistralLoader(
                base_url=self.kwargs.get("MISTRAL_OCR_API_BASE_URL"),
                api_key=self.kwargs.get("MISTRAL_OCR_API_KEY"),
                file_path=file_path,
            )
        else:
            if file_ext == "pdf":
                loader = PyPDFLoader(
                    file_path, extract_images=self.kwargs.get("PDF_EXTRACT_IMAGES")
                )
            elif file_ext == "csv":
                loader = CSVLoader(file_path, autodetect_encoding=True)
            elif file_ext == "rst":
                loader = UnstructuredRSTLoader(file_path, mode="elements")
            elif file_ext == "xml":
                loader = UnstructuredXMLLoader(file_path)
            elif file_ext in ["htm", "html"]:
                loader = BSHTMLLoader(file_path, open_encoding="unicode_escape")
            elif file_ext == "md":
                loader = TextLoader(file_path, autodetect_encoding=True)
            elif file_content_type == "application/epub+zip":
                loader = UnstructuredEPubLoader(file_path)
            elif (
                file_content_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                or file_ext == "docx"
            ):
                loader = Docx2txtLoader(file_path)
            elif file_content_type in [
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ] or file_ext in ["xls", "xlsx"]:
                loader = UnstructuredExcelLoader(file_path)
            elif file_content_type in [
                "application/vnd.ms-powerpoint",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ] or file_ext in ["ppt", "pptx"]:
                loader = UnstructuredPowerPointLoader(file_path)
            elif file_ext == "msg":
                loader = OutlookMessageLoader(file_path)
            elif file_ext == "odt":
                loader = UnstructuredODTLoader(file_path)
            elif self._is_text_file(file_ext, file_content_type):
                loader = TextLoader(file_path, autodetect_encoding=True)
            else:
                loader = TextLoader(file_path, autodetect_encoding=True)

        return loader
```

---

backend/open_webui/routers/retrieval.py — process_file (Entry-Point für Uploads, ruft Loader & Vektorisierung)

```python
@router.post("/process/file")
def process_file(
    request: Request,
    form_data: ProcessFileForm,
    user=Depends(get_verified_user),
):
    if user.role == "admin":
        file = Files.get_file_by_id(form_data.file_id)
    else:
        file = Files.get_file_by_id_and_user_id(form_data.file_id, user.id)

    if file:
        try:

            collection_name = form_data.collection_name

            if collection_name is None:
                collection_name = f"file-{file.id}"

            if form_data.content:
                # Update the content in the file
                # Usage: /files/{file_id}/data/content/update, /files/ (audio file upload pipeline)

                try:
                    # /files/{file_id}/data/content/update
                    VECTOR_DB_CLIENT.delete_collection(
                        collection_name=f"file-{file.id}"
                    )
                except:
                    # Audio file upload pipeline
                    pass

                docs = [
                    Document(
                        page_content=form_data.content.replace("<br/>", "\n"),
                        metadata={
                            **file.meta,
                            "name": file.filename,
                            "created_by": file.user_id,
                            "file_id": file.id,
                            "source": file.filename,
                        },
                    )
                ]

                text_content = form_data.content
            elif form_data.collection_name:
                # Check if the file has already been processed and save the content
                # Usage: /knowledge/{id}/file/add, /knowledge/{id}/file/update

                result = VECTOR_DB_CLIENT.query(
                    collection_name=f"file-{file.id}", filter={"file_id": file.id}
                )

                if result is not None and len(result.ids[0]) > 0:
                    docs = [
                        Document(
                            page_content=result.documents[0][idx],
                            metadata=result.metadatas[0][idx],
                        )
                        for idx, id in enumerate(result.ids[0])
                    ]
                else:
                    docs = [
                        Document(
                            page_content=file.data.get("content", ""),
                            metadata={
                                **file.meta,
                                "name": file.filename,
                                "created_by": file.user_id,
                                "file_id": file.id,
                                "source": file.filename,
                            },
                        )
                    ]

                text_content = file.data.get("content", "")
            else:
                # Process the file and save the content
                # Usage: /files/
                file_path = file.path
                if file_path:
                    file_path = Storage.get_file(file_path)
                    loader = Loader(
                        engine=request.app.state.config.CONTENT_EXTRACTION_ENGINE,
                        user=user,
                        DATALAB_MARKER_API_KEY=request.app.state.config.DATALAB_MARKER_API_KEY,
                        DATALAB_MARKER_API_BASE_URL=request.app.state.config.DATALAB_MARKER_API_BASE_URL,
                        DATALAB_MARKER_ADDITIONAL_CONFIG=request.app.state.config.DATALAB_MARKER_ADDITIONAL_CONFIG,
                        DATALAB_MARKER_SKIP_CACHE=request.app.state.config.DATALAB_MARKER_SKIP_CACHE,
                        DATALAB_MARKER_FORCE_OCR=request.app.state.config.DATALAB_MARKER_FORCE_OCR,
                        DATALAB_MARKER_PAGINATE=request.app.state.config.DATALAB_MARKER_PAGINATE,
                        DATALAB_MARKER_STRIP_EXISTING_OCR=request.app.state.config.DATALAB_MARKER_STRIP_EXISTING_OCR,
                        DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION=request.app.state.config.DATALAB_MARKER_DISABLE_IMAGE_EXTRACTION,
                        DATALAB_MARKER_FORMAT_LINES=request.app.state.config.DATALAB_MARKER_FORMAT_LINES,
                        DATALAB_MARKER_USE_LLM=request.app.state.config.DATALAB_MARKER_USE_LLM,
                        DATALAB_MARKER_OUTPUT_FORMAT=request.app.state.config.DATALAB_MARKER_OUTPUT_FORMAT,
                        EXTERNAL_DOCUMENT_LOADER_URL=request.app.state.config.EXTERNAL_DOCUMENT_LOADER_URL,
                        EXTERNAL_DOCUMENT_LOADER_API_KEY=request.app.state.config.EXTERNAL_DOCUMENT_LOADER_API_KEY,
                        TIKA_SERVER_URL=request.app.state.config.TIKA_SERVER_URL,
                        DOCLING_SERVER_URL=request.app.state.config.DOCLING_SERVER_URL,
                        DOCLING_PARAMS={
                            "do_ocr": request.app.state.config.DOCLING_DO_OCR,
                            "force_ocr": request.app.state.config.DOCLING_FORCE_OCR,
                            "ocr_engine": request.app.state.config.DOCLING_OCR_ENGINE,
                            "ocr_lang": request.app.state.config.DOCLING_OCR_LANG,
                            "pdf_backend": request.app.state.config.DOCLING_PDF_BACKEND,
                            "table_mode": request.app.state.config.DOCLING_TABLE_MODE,
                            "pipeline": request.app.state.config.DOCLING_PIPELINE,
                            "do_picture_description": request.app.state.config.DOCLING_DO_PICTURE_DESCRIPTION,
                            "picture_description_mode": request.app.state.config.DOCLING_PICTURE_DESCRIPTION_MODE,
                            "picture_description_local": request.app.state.config.DOCLING_PICTURE_DESCRIPTION_LOCAL,
                            "picture_description_api": request.app.state.config.DOCLING_PICTURE_DESCRIPTION_API,
                            **request.app.state.config.DOCLING_PARAMS,
                        },
                        PDF_EXTRACT_IMAGES=request.app.state.config.PDF_EXTRACT_IMAGES,
                        DOCUMENT_INTELLIGENCE_ENDPOINT=request.app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT,
                        DOCUMENT_INTELLIGENCE_KEY=request.app.state.config.DOCUMENT_INTELLIGENCE_KEY,
                        MISTRAL_OCR_API_BASE_URL=request.app.state.config.MISTRAL_OCR_API_BASE_URL,
                        MISTRAL_OCR_API_KEY=request.app.state.config.MISTRAL_OCR_API_KEY,
                        MINERU_API_MODE=request.app.state.config.MINERU_API_MODE,
                        MINERU_API_URL=request.app.state.config.MINERU_API_URL,
                        MINERU_API_KEY=request.app.state.config.MINERU_API_KEY,
                        MINERU_PARAMS=request.app.state.config.MINERU_PARAMS,
                    )
                    docs = loader.load(
                        file.filename, file.meta.get("content_type"), file_path
                    )

                    docs = [
                        Document(
                            page_content=doc.page_content,
                            metadata={
                                **filter_metadata(doc.metadata),
                                "name": file.filename,
                                "created_by": file.user_id,
                                "file_id": file.id,
                                "source": file.filename,
                            },
                        )
                        for doc in docs
                    ]
                else:
                    docs = [
                        Document(
                            page_content=file.data.get("content", ""),
                            metadata={
                                **file.meta,
                                "name": file.filename,
                                "created_by": file.user_id,
                                "file_id": file.id,
                                "source": file.filename,
                            },
                        )
                    ]
                text_content = " ".join([doc.page_content for doc in docs])

            log.debug(f"text_content: {text_content}")
            Files.update_file_data_by_id(
                file.id,
                {"content": text_content},
            )
            hash = calculate_sha256_string(text_content)
            Files.update_file_hash_by_id(file.id, hash)

            if request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL:
                Files.update_file_data_by_id(file.id, {"status": "completed"})
                return {
                    "status": True,
                    "collection_name": None,
                    "filename": file.filename,
                    "content": text_content,
                }
            else:
                try:
                    result = save_docs_to_vector_db(
                        request,
                        docs=docs,
                        collection_name=collection_name,
                        metadata={
                            "file_id": file.id,
                            "name": file.filename,
                            "hash": hash,
                        },
                        add=(True if form_data.collection_name else False),
                        user=user,
                    )
                    log.info(f"added {len(docs)} items to collection {collection_name}")

                    if result:
                        Files.update_file_metadata_by_id(
                            file.id,
                            {
                                "collection_name": collection_name,
                            },
                        )

                        Files.update_file_data_by_id(
                            file.id,
                            {"status": "completed"},
                        )

                        return {
                            "status": True,
                            "collection_name": collection_name,
                            "filename": file.filename,
                            "content": text_content,
                        }
                    else:
                        raise Exception("Error saving document to vector database")
                except Exception as e:
                    raise e

        except Exception as e:
            log.exception(e)
            Files.update_file_data_by_id(
                file.id,
                {"status": "failed"},
            )

            if "No pandoc was found" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e),
                )

    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.NOT_FOUND
        )
```

---

backend/open_webui/routers/retrieval.py — save_docs_to_vector_db (Chunken, Embeddings, Persistenz)

```python
def save_docs_to_vector_db(
    request: Request,
    docs,
    collection_name,
    metadata: Optional[dict] = None,
    overwrite: bool = False,
    split: bool = True,
    add: bool = False,
    user=None,
) -> bool:
    def _get_docs_info(docs: list[Document]) -> str:
        docs_info = set()

        # Trying to select relevant metadata identifying the document.
        for doc in docs:
            metadata = getattr(doc, "metadata", {})
            doc_name = metadata.get("name", "")
            if not doc_name:
                doc_name = metadata.get("title", "")
            if not doc_name:
                doc_name = metadata.get("source", "")
            if doc_name:
                docs_info.add(doc_name)

        return ", ".join(docs_info)

    log.info(
        f"save_docs_to_vector_db: document {_get_docs_info(docs)} {collection_name}"
    )

    # Check if entries with the same hash (metadata.hash) already exist
    if metadata and "hash" in metadata:
        result = VECTOR_DB_CLIENT.query(
            collection_name=collection_name,
            filter={"hash": metadata["hash"]},
        )

        if result is not None:
            existing_doc_ids = result.ids[0]
            if existing_doc_ids:
                log.info(f"Document with hash {metadata['hash']} already exists")
                raise ValueError(ERROR_MESSAGES.DUPLICATE_CONTENT)

    if split:
        if request.app.state.config.TEXT_SPLITTER in ["", "character"]:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=request.app.state.config.CHUNK_SIZE,
                chunk_overlap=request.app.state.config.CHUNK_OVERLAP,
                add_start_index=True,
            )
            docs = text_splitter.split_documents(docs)
        elif request.app.state.config.TEXT_SPLITTER == "token":
            log.info(
                f"Using token text splitter: {request.app.state.config.TIKTOKEN_ENCODING_NAME}"
            )

            tiktoken.get_encoding(str(request.app.state.config.TIKTOKEN_ENCODING_NAME))
            text_splitter = TokenTextSplitter(
                encoding_name=str(request.app.state.config.TIKTOKEN_ENCODING_NAME),
                chunk_size=request.app.state.config.CHUNK_SIZE,
                chunk_overlap=request.app.state.config.CHUNK_OVERLAP,
                add_start_index=True,
            )
            docs = text_splitter.split_documents(docs)
        elif request.app.state.config.TEXT_SPLITTER == "markdown_header":
            log.info("Using markdown header text splitter")

            # Define headers to split on - covering most common markdown header levels
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
                ("#####", "Header 5"),
                ("######", "Header 6"),
            ]

            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False,  # Keep headers in content for context
            )

            md_split_docs = []
            for doc in docs:
                md_header_splits = markdown_splitter.split_text(doc.page_content)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=request.app.state.config.CHUNK_SIZE,
                    chunk_overlap=request.app.state.config.CHUNK_OVERLAP,
                    add_start_index=True,
                )
                md_header_splits = text_splitter.split_documents(md_header_splits)

                # Convert back to Document objects, preserving original metadata
                for split_chunk in md_header_splits:
                    headings_list = []
                    # Extract header values in order based on headers_to_split_on
                    for _, header_meta_key_name in headers_to_split_on:
                        if header_meta_key_name in split_chunk.metadata:
                            headings_list.append(
                                split_chunk.metadata[header_meta_key_name]
                            )

                    md_split_docs.append(
                        Document(
                            page_content=split_chunk.page_content,
                            metadata={**doc.metadata, "headings": headings_list},
                        )
                    )

            docs = md_split_docs
        else:
            raise ValueError(ERROR_MESSAGES.DEFAULT("Invalid text splitter"))

    chunk_previews = []
    for doc in docs:
        preview = doc.page_content.replace("\n", " ")
        chunk_previews.append(preview)

    append_chunks_to_markdown(chunk_previews)

    if len(docs) == 0:
        raise ValueError(ERROR_MESSAGES.EMPTY_CONTENT)

    texts = [doc.page_content for doc in docs]
    metadatas = [
        {
            **doc.metadata,
            **(metadata if metadata else {}),
            "embedding_config": {
                "engine": request.app.state.config.RAG_EMBEDDING_ENGINE,
                "model": request.app.state.config.RAG_EMBEDDING_MODEL,
            },
        }
        for doc in docs
    ]

    try:
        if VECTOR_DB_CLIENT.has_collection(collection_name=collection_name):
            log.info(f"collection {collection_name} already exists")

            if overwrite:
                VECTOR_DB_CLIENT.delete_collection(collection_name=collection_name)
                log.info(f"deleting existing collection {collection_name}")
            elif add is False:
                log.info(
                    f"collection {collection_name} already exists, overwrite is False and add is False"
                )
                return True

        log.info(f"generating embeddings for {collection_name}")
        embedding_function = get_embedding_function(
            request.app.state.config.RAG_EMBEDDING_ENGINE,
            request.app.state.config.RAG_EMBEDDING_MODEL,
            request.app.state.ef,
            (
                request.app.state.config.RAG_OPENAI_API_BASE_URL
                if request.app.state.config.RAG_EMBEDDING_ENGINE == "openai"
                else (
                    request.app.state.config.RAG_OLLAMA_BASE_URL
                    if request.app.state.config.RAG_EMBEDDING_ENGINE == "ollama"
                    else request.app.state.config.RAG_AZURE_OPENAI_BASE_URL
                )
            ),
            (
                request.app.state.config.RAG_OPENAI_API_KEY
                if request.app.state.config.RAG_EMBEDDING_ENGINE == "openai"
                else (
                    request.app.state.config.RAG_OLLAMA_API_KEY
                    if request.app.state.config.RAG_EMBEDDING_ENGINE == "ollama"
                    else request.app.state.config.RAG_AZURE_OPENAI_API_KEY
                )
            ),
            request.app.state.config.RAG_EMBEDDING_BATCH_SIZE,
            azure_api_version=(
                request.app.state.config.RAG_AZURE_OPENAI_API_VERSION
                if request.app.state.config.RAG_EMBEDDING_ENGINE == "azure_openai"
                else None
            ),
        )

        embeddings = embedding_function(
            list(map(lambda x: x.replace("\n", " "), texts)),
            prefix=RAG_EMBEDDING_CONTENT_PREFIX,
            user=user,
        )
        log.info(f"embeddings generated {len(embeddings)} for {len(texts)} items")

        items = [
            {
                "id": str(uuid.uuid4()),
                "text": text,
                "vector": embeddings[idx],
                "metadata": metadatas[idx],
            }
            for idx, text in enumerate(texts)
        ]

        log.info(f"adding to collection {collection_name}")
        VECTOR_DB_CLIENT.insert(
            collection_name=collection_name,
            items=items,
        )

        log.info(f"added {len(items)} items to collection {collection_name}")
        return True
    except Exception as e:
        log.exception(e)
        raise e
```

---

backend/open_webui/retrieval/utils.py — get_embedding_function (Bau der Embedding-Funktion nach Engine)

```python
def get_embedding_function(
    embedding_engine,
    embedding_model,
    embedding_function,
    url,
    key,
    embedding_batch_size,
    azure_api_version=None,
):
    if embedding_engine == "":
        return lambda query, prefix=None, user=None: embedding_function.encode(
            query, **({"prompt": prefix} if prefix else {})
        ).tolist()
    elif embedding_engine in ["ollama", "openai", "azure_openai"]:
        func = lambda query, prefix=None, user=None: generate_embeddings(
            engine=embedding_engine,
            model=embedding_model,
            text=query,
            prefix=prefix,
            url=url,
            key=key,
            user=user,
            azure_api_version=azure_api_version,
        )

        def generate_multiple(query, prefix, user, func):
            if isinstance(query, list):
                embeddings = []
                for i in range(0, len(query), embedding_batch_size):
                    batch_embeddings = func(
                        query[i : i + embedding_batch_size],
                        prefix=prefix,
                        user=user,
                    )

                    if isinstance(batch_embeddings, list):
                        embeddings.extend(batch_embeddings)
                return embeddings
            else:
                return func(query, prefix, user)

        return lambda query, prefix=None, user=None: generate_multiple(
            query, prefix, user, func
        )
    else:
        raise ValueError(f"Unknown embedding engine: {embedding_engine}")
```

---

backend/open_webui/retrieval/utils.py — generate_openai/azure/ollama_batch_embeddings & generate_embeddings (HTTP-Aufrufe)

```python
def generate_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str = "https://api.openai.com/v1",
    key: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"generate_openai_batch_embeddings:model {model} batch size: {len(texts)}"
        )
        json_data = {"input": texts, "model": model}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        r = requests.post(
            f"{url}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                **(
                    {
                        "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                    }
                    if ENABLE_FORWARD_USER_INFO_HEADERS and user
                    else {}
                ),
            },
            json=json_data,
        )
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            return [elem["embedding"] for elem in data["data"]]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        log.exception(f"Error generating openai batch embeddings: {e}")
        return None


def generate_azure_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str,
    key: str = "",
    version: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"generate_azure_openai_batch_embeddings:deployment {model} batch size: {len(texts)}"
        )
        json_data = {"input": texts}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        url = f"{url}/openai/deployments/{model}/embeddings?api-version={version}"

        for _ in range(5):
            r = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "api-key": key,
                    **(
                        {
                            "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                            "X-OpenWebUI-User-Id": user.id,
                            "X-OpenWebUI-User-Email": user.email,
                            "X-OpenWebUI-User-Role": user.role,
                        }
                        if ENABLE_FORWARD_USER_INFO_HEADERS and user
                        else {}
                    ),
                },
                json=json_data,
            )
            if r.status_code == 429:
                retry = float(r.headers.get("Retry-After", "1"))
                time.sleep(retry)
                continue
            r.raise_for_status()
            data = r.json()
            if "data" in data:
                return [elem["embedding"] for elem in data["data"]]
            else:
                raise Exception("Something went wrong :/")
        return None
    except Exception as e:
        log.exception(f"Error generating azure openai batch embeddings: {e}")
        return None


def generate_ollama_batch_embeddings(
    model: str,
    texts: list[str],
    url: str,
    key: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"generate_ollama_batch_embeddings:model {model} batch size: {len(texts)}"
        )
        json_data = {"input": texts, "model": model}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        r = requests.post(
            f"{url}/api/embed",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                **(
                    {
                        "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                    }
                    if ENABLE_FORWARD_USER_INFO_HEADERS
                    else {}
                ),
            },
            json=json_data,
        )
        r.raise_for_status()
        data = r.json()

        if "embeddings" in data:
            return data["embeddings"]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        log.exception(f"Error generating ollama batch embeddings: {e}")
        return None


def generate_embeddings(
    engine: str,
    model: str,
    text: Union[str, list[str]],
    prefix: Union[str, None] = None,
    **kwargs,
):
    url = kwargs.get("url", "")
    key = kwargs.get("key", "")
    user = kwargs.get("user")

    if prefix is not None and RAG_EMBEDDING_PREFIX_FIELD_NAME is None:
        if isinstance(text, list):
            text = [f"{prefix}{text_element}" for text_element in text]
        else:
            text = f"{prefix}{text}"

    if engine == "ollama":
        embeddings = generate_ollama_batch_embeddings(
            **{
                "model": model,
                "texts": text if isinstance(text, list) else [text],
                "url": url,
                "key": key,
                "prefix": prefix,
                "user": user,
            }
        )
        return embeddings[0] if isinstance(text, str) else embeddings
    elif engine == "openai":
        embeddings = generate_openai_batch_embeddings(
            model, text if isinstance(text, list) else [text], url, key, prefix, user
        )
        return embeddings[0] if isinstance(text, str) else embeddings
    elif engine == "azure_openai":
        azure_api_version = kwargs.get("azure_api_version", "")
        embeddings = generate_azure_openai_batch_embeddings(
            model,
            text if isinstance(text, list) else [text],
            url,
            key,
            azure_api_version,
            prefix,
            user,
        )
        return embeddings[0] if isinstance(text, str) else embeddings
```

---

backend/open_webui/retrieval/utils.py — VectorSearchRetriever & query_doc_with_hybrid_search (Retrieval, Hybrid mit BM25)

```python
class VectorSearchRetriever(BaseRetriever):
    collection_name: Any
    embedding_function: Any
    top_k: int

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        result = VECTOR_DB_CLIENT.search(
            collection_name=self.collection_name,
            vectors=[self.embedding_function(query, RAG_EMBEDDING_QUERY_PREFIX)],
            limit=self.top_k,
        )

        ids = result.ids[0]
        metadatas = result.metadatas[0]
        documents = result.documents[0]

        results = []
        for idx in range(len(ids)):
            results.append(
                Document(
                    metadata=metadatas[idx],
                    page_content=documents[idx],
                )
            )
        return results


def query_doc(
    collection_name: str, query_embedding: list[float], k: int, user: UserModel = None
):
    try:
        log.debug(f"query_doc:doc {collection_name}")
        result = VECTOR_DB_CLIENT.search(
            collection_name=collection_name,
            vectors=[query_embedding],
            limit=k,
        )

        if result:
            log.info(f"query_doc:result {result.ids} {result.metadatas}")

        return result
    except Exception as e:
        log.exception(f"Error querying doc {collection_name} with limit {k}: {e}")
        raise e


def get_doc(collection_name: str, user: UserModel = None):
    try:
        log.debug(f"get_doc:doc {collection_name}")
        result = VECTOR_DB_CLIENT.get(collection_name=collection_name)

        if result:
            log.info(f"query_doc:result {result.ids} {result.metadatas}")

        return result
    except Exception as e:
        log.exception(f"Error getting doc {collection_name}: {e}")
        raise e


def query_doc_with_hybrid_search(
    collection_name: str,
    collection_result: GetResult,
    query: str,
    embedding_function,
    k: int,
    reranking_function,
    k_reranker: int,
    r: float,
    hybrid_bm25_weight: float,
) -> dict:
    try:
        if (
            not collection_result
            or not hasattr(collection_result, "documents")
            or not collection_result.documents
            or len(collection_result.documents) == 0
            or not collection_result.documents[0]
        ):
            log.warning(f"query_doc_with_hybrid_search:no_docs {collection_name}")
            return {"documents": [], "metadatas": [], "distances": []}

        log.debug(f"query_doc_with_hybrid_search:doc {collection_name}")

        bm25_retriever = BM25Retriever.from_texts(
            texts=collection_result.documents[0],
            metadatas=collection_result.metadatas[0],
        )
        bm25_retriever.k = k

        vector_search_retriever = VectorSearchRetriever(
            collection_name=collection_name,
            embedding_function=embedding_function,
            top_k=k,
        )

        if hybrid_bm25_weight <= 0:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_search_retriever], weights=[1.0]
            )
        elif hybrid_bm25_weight >= 1:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever], weights=[1.0]
            )
        else:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_search_retriever],
                weights=[hybrid_bm25_weight, 1.0 - hybrid_bm25_weight],
            )

        compressor = RerankCompressor(
            embedding_function=embedding_function,
            top_n=k_reranker,
            reranking_function=reranking_function,
            r_score=r,
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )

        result = compression_retriever.invoke(query)

        distances = [d.metadata.get("score") for d in result]
        documents = [d.page_content for d in result]
        metadatas = [d.metadata for d in result]

        # retrieve only min(k, k_reranker) items, sort and cut by distance if k < k_reranker
        if k < k_reranker:
            sorted_items = sorted(
                zip(distances, metadatas, documents), key=lambda x: x[0], reverse=True
            )
            sorted_items = sorted_items[:k]

            if sorted_items:
                distances, documents, metadatas = map(list, zip(*sorted_items))
            else:
                distances, documents, metadatas = [], [], []

        result = {
            "distances": [distances],
            "documents": [documents],
            "metadatas": [metadatas],
        }

        log.info(
            "query_doc_with_hybrid_search:result "
            + f'{result["metadatas"]} {result["distances"]}'
        )
        return result
    except Exception as e:
        log.exception(f"Error querying doc {collection_name} with hybrid search: {e}")
        raise e
```

---

backend/open_webui/routers/retrieval.py — query_doc_handler/query_collection_handler (API-Einstieg für Retrieval)

```python
@router.post("/query/doc")
def query_doc_handler(
    request: Request,
    form_data: QueryDocForm,
    user=Depends(get_verified_user),
):
    try:
        if request.app.state.config.ENABLE_RAG_HYBRID_SEARCH and (
            form_data.hybrid is None or form_data.hybrid
        ):
            collection_results = {}
            collection_results[form_data.collection_name] = VECTOR_DB_CLIENT.get(
                collection_name=form_data.collection_name
            )
            return query_doc_with_hybrid_search(
                collection_name=form_data.collection_name,
                collection_result=collection_results[form_data.collection_name],
                query=form_data.query,
                embedding_function=lambda query, prefix: request.app.state.EMBEDDING_FUNCTION(
                    query, prefix=prefix, user=user
                ),
                k=form_data.k if form_data.k else request.app.state.config.TOP_K,
                reranking_function=(
                    (
                        lambda sentences: request.app.state.RERANKING_FUNCTION(
                            sentences, user=user
                        )
                    )
                    if request.app.state.RERANKING_FUNCTION
                    else None
                ),
                k_reranker=form_data.k_reranker
                or request.app.state.config.TOP_K_RERANKER,
                r=(
                    form_data.r
                    if form_data.r
                    else request.app.state.config.RELEVANCE_THRESHOLD
                ),
                hybrid_bm25_weight=(
                    form_data.hybrid_bm25_weight
                    if form_data.hybrid_bm25_weight
                    else request.app.state.config.HYBRID_BM25_WEIGHT
                ),
                user=user,
            )
        else:
            return query_doc(
                collection_name=form_data.collection_name,
                query_embedding=request.app.state.EMBEDDING_FUNCTION(
                    form_data.query, prefix=RAG_EMBEDDING_QUERY_PREFIX, user=user
                ),
                k=form_data.k if form_data.k else request.app.state.config.TOP_K,
                user=user,
            )
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@router.post("/query/collection")
def query_collection_handler(
    request: Request,
    form_data: QueryCollectionsForm,
    user=Depends(get_verified_user),
):
    try:
        if request.app.state.config.ENABLE_RAG_HYBRID_SEARCH and (
            form_data.hybrid is None or form_data.hybrid
        ):
            return query_collection_with_hybrid_search(
                collection_names=form_data.collection_names,
                queries=[form_data.query],
                embedding_function=lambda query, prefix: request.app.state.EMBEDDING_FUNCTION(
                    query, prefix=prefix, user=user
                ),
                k=form_data.k if form_data.k else request.app.state.config.TOP_K,
                reranking_function=(
                    (
                        lambda sentences: request.app.state.RERANKING_FUNCTION(
                            sentences, user=user
                        )
                    )
                    if request.app.state.RERANKING_FUNCTION
                    else None
                ),
                k_reranker=form_data.k_reranker
                or request.app.state.config.TOP_K_RERANKER,
                r=(
                    form_data.r
                    if form_data.r
                    else request.app.state.config.RELEVANCE_THRESHOLD
                ),
                hybrid_bm25_weight=(
                    form_data.hybrid_bm25_weight
                    if form_data.hybrid_bm25_weight
                    else request.app.state.config.HYBRID_BM25_WEIGHT
                ),
            )
        else:
            return query_collection(
                collection_names=form_data.collection_names,
                queries=[form_data.query],
                embedding_function=lambda query, prefix: request.app.state.EMBEDDING_FUNCTION(
                    query, prefix=prefix, user=user
                ),
                k=form_data.k if form_data.k else request.app.state.config.TOP_K,
            )

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )
```
