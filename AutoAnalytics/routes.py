import os
from flask import render_template, Blueprint, flash, send_file, redirect
from flask.helpers import url_for
import pandas as pd
from AutoAnalytics.forms import UploadDataset
from AutoAnalytics.utils import save_file
from AutoAnalytics.analysis import AutoAnalytics

main = Blueprint('main', __name__)

@main.route("/", methods = ['GET', 'POST'])
@main.route("/home", methods = ['GET', 'POST'])
def home():
    form = UploadDataset()
    if form.validate_on_submit():
        flash('The file has been uploaded. The results will be returned shortly.', 'success')
        file_name = save_file(form.dataset.data)

        analytics = AutoAnalytics(path='datasets/'+file_name,
         dependent_variable='dependent_variable',
         train_test_column='train_test')
        analytics.fit_models()
        analytics.make_prediction()
        analytics.save_prediction()        

        return redirect(url_for('main.download_file', file_name = file_name))  
    
    return render_template('home.html', form=form)

@main.route('/download/<string:file_name>')
def download_file(file_name):
    return send_file('datasets/'+file_name, as_attachment=True)

@main.route("/about")
def about():
    return render_template('about.html', title = 'About')
