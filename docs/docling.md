# Docling Document Extraction

Open WebUI can send files to a **Docling Serve** instance instead of the default text loader, which unlocks the same GPU-accelerated document-conversion pipeline described in the [official walkthrough](https://docs.openwebui.com/features/rag/document-extraction/docling).

## 1. Run the GPU container

Docling exposes a REST API on port `5001`. The GPU-ready image that you already pulled can be started with:

```bash
docker run --gpus all -p 5001:5001 -e DOCLING_SERVE_ENABLE_UI=true quay.io/docling-project/docling-serve-cu124
```

Once the server is up, http://localhost:5001/docs shows the OpenAPI schema and http://localhost:5001/ui exposes the optional UI.

## 2. Point Open WebUI to Docling

In the **Document settings** tab, switch `Content Extraction Engine` to `Docling`, then paste `http://localhost:5001` (or whatever host/port your container listens on) into the Docling Server URL field. Open WebUI stores the URL in `DOCLING_SERVER_URL`, so you can also predefine it via `.env` (see `backend/open_webui/config.py` for the exact environment variable name).

## 3. Fine-tune the request payload

The backend sends a subset of the Docling API options from the admin UI, plus whatever JSON you paste into the `Docling Parameters` text area. The payload shown in `backend/open_webui/retrieval/loaders/main.py` is merged with your JSON and sent as form data to `/v1/convert/file`.

Here is a typical `DOCLING_PARAMS` object you can paste into the textarea (strings in the UI must be valid JSON):

```json
{
  "do_ocr": true,
  "force_ocr": false,
  "ocr_engine": "tesseract",
  "ocr_lang": ["eng", "fra", "deu", "spa"],
  "pdf_backend": "dlparse_v4",
  "table_mode": "accurate",
  "pipeline": "standard",
  "do_picture_description": false
}
```

If you need image captions or the `vlm` pipeline, add the Hugging Face repo that Docling should load instead of letting it guess:

```json
{
  "do_picture_description": true,
  "picture_description_local": {
    "repo_id": "HuggingFaceTB/SmolVLM-256M-Instruct"
  }
}
```

The `repo_id` must be in the `namespace/repo` form requested by Docling. Passing the local cache path (`/opt/app-root/src/.cache/docling/models/HuggingFaceTB--SmolVLM-256M-Instruct`) triggers the error you saw; set the Hugging Face identifier instead so the underlying `huggingface_hub` call succeeds.

## 4. Troubleshooting

- **“Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/opt/...HuggingFaceTB--SmolVLM-256M-Instruct'”** – this means Docling tried to pass a filesystem path to `huggingface_hub.snapshot_download`. Make sure your `picture_description_local`/`vlm_pipeline_model_local` entry uses the Hugging Face repo identifier (`HuggingFaceTB/SmolVLM-256M-Instruct`) rather than the container cache path. You can paste the entire JSON snippet above into the `Docling Parameters` textarea.
- **404 response from `/v1/convert/file`** – check the Docling logs for the same `repo_id` error, restart the container, and confirm that the server URL in Open WebUI points to the running container.

## 5. Recap

1. Start `docling-serve` (CPU and GPU variants behave the same from the HTTP API).
2. Set `DOCLING_SERVER_URL` and choose the Docling engine in the admin UI.
3. Tweak `DOCLING_PARAMS` to control OCR, pipelines, and optional vision-language models. When Docling complains about `repo_id`, switch to the Hugging Face identifier instead of a local path.

The same JSON that you paste into the doc parameters textarea is the data forwarded to the Docling `/v1/convert/file` endpoint, so you can copy/paste examples from the [official Docling usage docs](https://docs.openwebui.com/features/rag/document-extraction/docling) or from the `docling-serve` repository.
