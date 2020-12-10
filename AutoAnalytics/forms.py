from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError

class UploadDataset(FlaskForm):
    dataset = FileField('Upload the dataset in a csv format.', validators=[FileAllowed(['csv'])])
    submit = SubmitField('Upload')
