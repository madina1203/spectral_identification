import os
import sys
from matchms.importing import load_from_msp
from matchms.exporting import save_as_mgf
from matchms.Pipeline import Pipeline, load_workflow_from_yaml_file
import yaml



def main():

    config_path = "/Users/madinabekbergenova/PycharmProjects/pythonProject/config/matchms_pipeline_settings.yaml"
    reference_library_path = "/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/spectral_libraries/hr_msms_nist_all.MSP"
    cleaned_library_path = "/Users/madinabekbergenova/Desktop/phd_data/methods_by_orbitraps/spectral_libraries/cleaned_hr_msms_nist_all.mgf"
    workflow = load_workflow_from_yaml_file(config_path)
    pipeline = Pipeline(workflow)
    pipeline.logging_file = "my_pipeline.log"
    pipeline.logging_level = "ERROR"

    pipeline.run(query_files= reference_library_path, cleaned_query_file = cleaned_library_path)
    print("library is cleaned")



if __name__ == '__main__':
    main()