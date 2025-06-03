import nltk
nltk.download('punkt')

from unstructured.partition.pdf import partition_pdf

output_path = "./content/"
file_path = output_path + 'Chapter 3_Waste_Reduction.pdf'


chunks = partition_pdf(
    filename=file_path,
    infer_table_structure=True,            
    strategy="hi_res",                     

    extract_image_block_types=["Image"],   

    extract_image_block_to_payload=True,  

    chunking_strategy="by_title",          
    max_characters=10000,                  
    combine_text_under_n_chars=2000,      
    new_after_n_chars=6000,
)

set([str(type(el)) for el in chunks])
chunks[3].metadata.orig_elements

elements = chunks[3].metadata.orig_elements
chunk_images = [el for el in elements if 'Image' in str(type(el))]
chunk_images[0].to_dict()

tables = []
texts = []

for chunk in chunks:
    if "Table" in str(type(chunk)):
        tables.append(chunk)

    if "CompositeElement" in str(type((chunk))):
        texts.append(chunk)

def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

images = get_images_base64(chunks)

import base64
from IPython.display import Image, display

def display_base64_image(base64_code):
    # Decode the base64 string to binary
    image_data = base64.b64decode(base64_code)
    # Display the image
    display(Image(data=image_data))

display_base64_image(images[0])

import pickle

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)