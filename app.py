from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
PDF_FOLDER = 'static/pdfs/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['PDF_FOLDER'] = PDF_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Maan lete hain ki tumhara segmentation model ek segmented image bana ke yahin save karta hai:
            segmented_filename = filename.split('.')[0] + '_segmented.png'
            segmented_path = os.path.join(app.config['RESULT_FOLDER'], segmented_filename)

            # Assume yeh values model se aa rahi hain
            grade = 2
            grade_name = "Moderate Non-Proliferative Diabetic Retinopathy"
            advice = "Schedule a comprehensive eye exam and monitor blood sugar levels."

            pdf_filename = filename.split('.')[0] + '_report.pdf'

            return render_template('result.html',
                                   original_image=url_for('static', filename='uploads/' + filename),
                                   segmented_filename = filename.split('.')[0] + '_segmented.png',
                                   grade=grade,
                                   grade_name=grade_name,
                                   advice=advice,
                                   pdf_url=url_for('static', filename='pdfs/' + pdf_filename)
                                   )
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
