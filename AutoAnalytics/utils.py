from datetime import datetime
import os
import secrets
from flask import current_app

def save_file(form_file):
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name, file_extension = os.path.splitext(form_file.filename)
    random_hex = secrets.token_hex(8)
    saved_file_name = timestamp + '_' + file_name + random_hex + file_extension
    saved_file_path = os.path.join(current_app.root_path, 'datasets', saved_file_name)
    form_file.save(saved_file_path)

    return saved_file_name