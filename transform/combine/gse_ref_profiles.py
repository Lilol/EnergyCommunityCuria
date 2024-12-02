from data_processing_pipeline.data_processing_pipeline import DataProcessingPipeline
from input.read import ReadGseDatabase
from output.write import WriteGseProfile
from transform.transform import TransformReferenceProfile

# ----------------------------------------------------------------------------
DataProcessingPipeline("gse_ref_profiles", workers=(
    ReadGseDatabase(),
    TransformReferenceProfile(),
    WriteGseProfile("gse_ref_profiles"))).execute()
