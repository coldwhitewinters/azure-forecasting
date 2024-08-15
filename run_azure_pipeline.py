from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.dsl import pipeline

from mldesigner import command_component, Input, Output
from src.preprocessing import prepare_m5_data
from src.hierarchical import build_hierarchy
from src.forecast import forecast
# import os


@command_component(
    name="prepare_data",
    version="2",
    display_name="Prepare data",
    description="Prepare M5 data",
    environment="azureml:azure-forecast-env@latest",
)
def prepare_data_component(
    input_dir: Input(type="uri_folder"),
    output_dir: Output(type="uri_folder")
):
    prepare_m5_data(input_dir, output_dir, max_series=None)


@command_component(
    name="build_hierarchy",
    version="1",
    display_name="Build hierarchy",
    description="Build hierarchy of time series",
    environment="azureml:azure-forecast-env@latest",
)
def build_hierarchy_component(
    input_dir: Input(type="uri_folder"),
    output_dir: Output(type="uri_folder")
):
    build_hierarchy(input_dir, output_dir)


@command_component(
    name="forecast",
    version="2",
    display_name="Forecast",
    description="Forecast M5 data",
    environment="azureml:azure-forecast-env@latest",
)
def forecast_component(
    input_dir: Input(type="uri_folder"),
    output_dir: Output(type="uri_folder")
):
    forecast(input_dir, output_dir)


@pipeline(
    name="forecasting_pipeline",
    version="3",
    display_name="Forecasting pipeline",
    description="Forecasting pipeline for the M5 data",
    experiment_name="forecasting_experiment",
    default_compute="forecast-compute-D32",
)
def forecasting_pipeline(input_dir):
    """Forecasting pipeline for the M5 data"""
    prepare_data_node = prepare_data_component(
        input_dir=input_dir, 
    )
    build_hierarchy_node = build_hierarchy_component(
        input_dir=prepare_data_node.outputs.output_dir
    )
    forecast_node = forecast_component(
        input_dir=build_hierarchy_node.outputs.output_dir
    )

    return {
        "processed_data": prepare_data_node.outputs.output_dir,
        "forecast": forecast_node.outputs.output_dir
    }


def main():
    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id="b42e58dd-01c0-4a33-882e-aeff6b561cd4",
        resource_group_name="azure-forecasting",
        workspace_name="azure-forecasting",
    )

    m5_data = ml_client.data.get(name="m5-data", version="1")
    pipeline_job = forecasting_pipeline(input_dir=m5_data)

    ml_client.jobs.create_or_update(pipeline_job)


if __name__ == "__main__":
    main()
