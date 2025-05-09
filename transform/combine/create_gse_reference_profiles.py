from data_processing_pipeline.data_processing_pipeline import DataProcessingPipeline
from io_operation.input.read import ReadGseDatabase
from io_operation.output.write import WriteGseProfile
from transform.transform import TransformReferenceProfile

# ----------------------------------------------------------------------------
DataProcessingPipeline("gse_ref_profiles", workers=(
    ReadGseDatabase(),
    TransformReferenceProfile(),
    WriteGseProfile("gse_ref_profiles"))).execute()
