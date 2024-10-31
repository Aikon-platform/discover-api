import dramatiq
from typing import Optional

from .const import SIM_QUEUE
from .lib.similarity import LoggedComputeSimilarity
from ..shared.utils.logging import notifying, TLogger, LoggerHelper


# @notifying TODO implement results return with notifying
@dramatiq.actor(time_limit=1000 * 60 * 60, max_retries=0, store_results=True, queue_name=SIM_QUEUE)
def compute_similarity(
    experiment_id: str,
    dataset: dict,
    parameters: Optional[dict] = None,
    notify_url: Optional[str] = None,
    tracking_url: Optional[str] = None,
    logger: TLogger = LoggerHelper,
):
    """
    Run similarity retrieval task on all pairs of documents

    Parameters:
    - experiment_id: the ID of the similarity retrieval task
    - dataset: dictionary containing the documents to be compared
    - parameters: a JSON object containing the task parameters (model)
    - notify_url: the URL to be called when the task is finished
    - logger: a logger object

    E.g. of dataset dict
    "documents": {
        "wit3_man186_anno181": "https://eida.obspm.fr/eida/wit3_man186_anno181/list/",
        "wit87_img87_anno87": "https://eida.obspm.fr/eida/wit87_img87_anno87/list/",
        "wit2_img2_anno2": "https://eida.obspm.fr/eida/wit2_img2_anno2/list/"
    }
    """

    similarity_task = LoggedComputeSimilarity(
        logger,
        experiment_id=experiment_id,
        dataset=dataset,
        parameters=parameters,
        notify_url=notify_url,
        tracking_url=tracking_url
    )
    similarity_task.run_task()
