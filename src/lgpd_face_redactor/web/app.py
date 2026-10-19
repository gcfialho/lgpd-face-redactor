"""FastAPI app: upload a photo or video, tick who stays visible, download the redacted copy.

    uv run --extra web lgpd-redact-web            # http://127.0.0.1:8000

Meant to run on the machine (or private server) of whoever handles the access request.
The UI loads nothing from third parties: no CDNs, fonts or analytics.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from lgpd_face_redactor import __version__
from lgpd_face_redactor.types import Style
from lgpd_face_redactor.web.jobs import JobStore, Status, UploadError, WebJob

HERE = Path(__file__).parent
templates = Jinja2Templates(directory=HERE / "templates")
NO_STORE = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}


def default_store() -> JobStore:
    from lgpd_face_redactor.vision import SFaceEmbedder, YuNetDetector

    return JobStore(
        detector_factory=YuNetDetector,
        embedder_factory=SFaceEmbedder,
        ttl_seconds=int(os.getenv("LGPD_REDACTOR_TTL_SECONDS", "3600")),
        max_upload_bytes=int(os.getenv("LGPD_REDACTOR_MAX_UPLOAD_MB", "500")) << 20,
    )


def create_app(store: JobStore | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store or default_store()
        yield
        app.state.store.shutdown()  # nothing outlives the server

    app = FastAPI(title="LGPD Face Redactor", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.middleware("http")
    async def privacy_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(NO_STORE)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; media-src 'self'; style-src 'self'; script-src 'self'"
        )
        return response

    def job_or_404(request: Request, job_id: str) -> WebJob:
        job = request.app.state.store.get(job_id)
        if job is None:
            raise HTTPException(404, "This job does not exist or has expired")
        return job

    @app.get("/")
    def index(request: Request, error: str | None = None):
        return templates.TemplateResponse(request, "index.html", {"styles": list(Style), "error": error})

    @app.post("/jobs")
    def create_job(
        request: Request,
        file: Annotated[UploadFile, File()],
        references: Annotated[list[UploadFile] | None, File()] = None,
        style: Annotated[Style, Form()] = Style.PIXELATE,
    ):
        refs = [(r.file, r.filename or "reference.jpg") for r in (references or []) if r.filename]
        try:
            job = request.app.state.store.create(file.file, file.filename or "upload", refs, style)
        except UploadError as exc:
            return templates.TemplateResponse(
                request, "index.html", {"styles": list(Style), "error": str(exc)}, status_code=400
            )
        return RedirectResponse(f"/jobs/{job.id}", status_code=303)

    @app.get("/jobs/{job_id}")
    def job_page(request: Request, job_id: str):
        job = job_or_404(request, job_id)
        return templates.TemplateResponse(request, "job.html", {"job": job, "Status": Status, "styles": list(Style)})

    @app.get("/api/jobs/{job_id}")
    def job_status(request: Request, job_id: str):
        job = job_or_404(request, job_id)
        return {"status": job.status, "progress": round(job.progress, 3), "message": job.message, "error": job.error}

    @app.get("/jobs/{job_id}/people/{person_id}.jpg")
    def thumbnail(request: Request, job_id: str, person_id: int):
        path = job_or_404(request, job_id).thumbnails.get(person_id)
        if path is None or not path.exists():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/jobs/{job_id}/render")
    def render(
        request: Request,
        job_id: str,
        keep: Annotated[list[int] | None, Form()] = None,
        style: Annotated[Style, Form()] = Style.PIXELATE,
    ):
        job = job_or_404(request, job_id)
        try:
            request.app.state.store.render(job, set(keep or []), style)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return RedirectResponse(f"/jobs/{job.id}", status_code=303)

    @app.get("/jobs/{job_id}/preview")
    def preview(request: Request, job_id: str):
        job = job_or_404(request, job_id)
        if job.preview is None or not job.preview.exists():
            raise HTTPException(404, "No browser preview available")
        return FileResponse(job.preview)

    @app.get("/jobs/{job_id}/download")
    def download(request: Request, job_id: str):
        job = job_or_404(request, job_id)
        if job.status is not Status.DONE or job.output is None:
            raise HTTPException(409, "Nothing to download yet")
        path = job.preview if job.preview and not job.info.is_image else job.output
        name = f"{Path(job.original_name).stem}-redacted{path.suffix if job.info.is_image else '.mp4'}"
        return FileResponse(path, filename=name)

    @app.get("/jobs/{job_id}/audit.json")
    def audit_record(request: Request, job_id: str):
        job = job_or_404(request, job_id)
        if job.record is None:
            raise HTTPException(409, "No audit record yet")
        name = f"{Path(job.original_name).stem}-redacted.audit.json"
        return Response(
            json.dumps(job.record, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    @app.post("/jobs/{job_id}/delete")
    def delete(request: Request, job_id: str):
        request.app.state.store.delete(job_id)
        return RedirectResponse("/?deleted=1", status_code=303)

    @app.get("/healthz")
    def health():
        return JSONResponse({"ok": True, "version": __version__})

    return app


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="lgpd-redact-web", description="Run the redaction web app locally.")
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 only inside a container or a private network")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
