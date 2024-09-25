import unittest
from pathlib import Path
import shutil

from ..lib.src.utils import path


class TestTraining(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.TMP_PATH = Path("/tmp/dti/")
        self.DATA_PATH = Path(__file__).parent / "data"

        path.RUNS_PATH = self.TMP_PATH / "runs"
        path.DATASETS_PATH = self.DATA_PATH / "datasets"
        path.CONFIGS_PATH = self.TMP_PATH / "configs"

        from api.app.dticlustering import training

        training.KMEANS_CONFIG_FILE = self.DATA_PATH / "kmeans-conf.yml"
        training.SPRITES_CONFIG_FILE = self.DATA_PATH / "sprites-conf.yml"
        training.RUNS_PATH = path.RUNS_PATH

    def tearDown(self) -> None:
        super().tearDown()
        shutil.rmtree(self.TMP_PATH)

    def test_run_kmeans_training(self):
        from .. import training

        clustering_id = "test_kmeans"
        dataset_id = "example"
        parameters = {
            "n_prototypes": 5,
            "transformation_sequence": "identity_color_affine_projective_tps",
        }
        self.assertIsNotNone(
            training.run_kmeans_training(clustering_id, dataset_id, parameters)
        )

    def test_run_sprites_training(self):
        from .. import training

        clustering_id = "test_clustering_id"
        dataset_id = "example"
        parameters = {
            "n_prototypes": 5,
            "transformation_sequence": "identity_color_affine_projective_tps",
        }
        self.assertIsNotNone(
            training.run_sprites_training(clustering_id, dataset_id, parameters)
        )
