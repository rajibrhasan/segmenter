import pandas as pd
import json
import re

'''This patch is located in Dhaka North City Corporation, within Badda Upazila of Dhaka District.
Dhaka North City Corporation has a population density of 10,000 people per square kilometer and a literacy rate of 90 percent.'''

def generate_metadata_backbone(row):
    description = f"This patch is located in {row['Location']}, within {row['ADM3_EN']} Upazila of {row['ADM2_EN']} District. "
    description += f"{row['Location']} has a population density of {row['Population Density']} people per square kilometer and a literacy rate of {row['Literacy']}."

    return description


def generate_union_metadata(file_path, sheet_name):
    unions_df = pd.read_excel(file_path, sheet_name=sheet_name)
    unions_df['metadata'] = unions_df.apply(generate_metadata_backbone, axis=1)
    unions_df['union_upazila'] =  unions_df['ADM4_EN'] + '_' + unions_df['ADM3_EN']
    union_dict = dict(zip(unions_df['union_upazila'], unions_df['metadata']))
    with open('data/metadata_backbone.json', 'w', encoding='utf-8') as f:
        json.dump(union_dict, f, ensure_ascii=False, indent=2)


union_file_path = 'data/union_info.xlsx'
sheet_name = 'Final'
generate_union_metadata(union_file_path, sheet_name)