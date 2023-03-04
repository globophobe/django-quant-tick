from diagrams import Cluster, Diagram
from diagrams.custom import Custom
from diagrams.gcp.compute import Run
from diagrams.gcp.devtools import Scheduler

with Diagram("", show=False):
    scheduler = Scheduler("On schedule")
    download = Custom("Download trade data", "./workflows.png")
    aggregate = Custom("Aggregate candlesticks", "./workflows.png")
    django = Custom("", "./django-pony.png", height="3", width="3")

    with Cluster("Serverless tasks"):
        serverless = [
            Run("Download task A"),
            Run("Download task B"),
            Run("Download task C"),
        ]

    scheduler >> download >> serverless >> aggregate >> django
