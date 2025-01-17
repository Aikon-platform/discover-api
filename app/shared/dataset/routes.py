"""
Routes for handling datasets

Allows to fetch and download content from a dataset
(mostly to avoid downloading IIIF manifests multiple times)
"""
from flask import Blueprint, jsonify, request, stream_with_context, url_for, Response
import os

from ..utils.fileutils import zip_on_the_fly, sanitize_str

from .dataset import Dataset, DATASETS_PATH
from .document import Document

blueprint = Blueprint("datasets", __name__, url_prefix="/datasets")


@blueprint.route("dataset/<uid>", methods=["GET"])
def dataset_info(uid):
    """
    Get the dataset information
    """
    dataset = Dataset(uid, load=True)
    return jsonify(dataset.to_dict(with_url=True))


@blueprint.route("document/<dtype>/<path:uid>", methods=["GET"])
def document_info(dtype, uid):
    """
    Get the document information
    """
    document = Document(uid, dtype)
    return jsonify(document.to_dict(with_url=True))


@blueprint.route("document/<dtype>/<path:uid>/download", methods=["GET"])
def document_download(dtype, uid):
    """
    Download the document
    """
    document = Document(uid, dtype)
    files = [
        ("images/" + im.path.name, im.path) for im in document.list_images()
    ] + [("mapping.json", document.mapping_path)]

    fname = f"{sanitize_str(document.uid)}.zip"

    return Response(
        stream_with_context(zip_on_the_fly(files)),
        mimetype="application/zip",
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )
