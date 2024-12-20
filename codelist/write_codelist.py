import pandas as pd

# Read the Excel file
excel_file_path = 'codelist.xlsx'
df = pd.read_excel(excel_file_path)

# Get the content of column A
column_a_data = df['证券代码']
# print(column_a_data)

# Write the content to a text file
text_file_path = 'codelist.txt'
with open(text_file_path, 'w') as text_file:
    for item in column_a_data:
        if '.BJ' not in item:
            if '.SH' in item:
                new_string = "1|" + item.replace(".SH", "")
                text_file.write(new_string + '\n')
            elif '.SZ' in item:
                new_string = "0|" + item.replace(".SZ", "")
                text_file.write(new_string + '\n')

print(f"Content of column A written to {text_file_path}")