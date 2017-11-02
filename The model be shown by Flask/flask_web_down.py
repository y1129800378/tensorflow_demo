# coding=utf-8
import sys
import test_modle_bingli
reload(sys)
sys.setdefaultencoding("utf-8")
import os
from flask import Flask,url_for,render_template,request,url_for,redirect,send_from_directory
from werkzeug import secure_filename

UPLOAD_FOLDER='/home/yyy/flask_pic/'
ALLOWED_EXTENSIONS=set(['txt','pdf','png','jpg','jpeg','gif','JPG'])

app=Flask(__name__)

app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

i=int(3)
filenames='a'

def inference(file_name):
    try:
        predictions, top_k, top_names = run_inference_on_image(file_name)
        print(predictions)
    except Exception as ex: 
        print(ex)
        return ""
    new_url = '/static/%s' % os.path.basename(file_name)
    image_tag = '<img src="%s"></img><p>'
    new_tag = image_tag % new_url
    format_string = ''
    for node_id, human_name in zip(top_k, top_names):
        score = predictions[node_id]
        format_string += '%s (score:%.5f)<BR>' % (human_name, score)
    ret_string = new_tag  + format_string + '<BR>' 
    return ret_string
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

@app.route('/',methods=['GET','POST'])
def upload_file():
    str_ret='''
    <!DOCTYPE html>
    <title>临时测试用</title>
    <h1>来喂一张照片吧</h1>
    <form action="/" method="POST" enctype="multipart/form-data">
    <input type="file" name="file" />
    <input type="submit" value="upload" />
    </form>
    '''
    str_ret2='''
    <!DOCTYPE html>
    <title>临时测试用</title>
    <h1>来喂一张照片吧</h1>
    <form action="/" method="POST" enctype="multipart/form-data">
    <input type="file" name="file" />
    <input type="submit" value="upload" />
    
    </form>
    '''
    if request.method=='POST':
        file=request.files['file']
        if file and allowed_file(file.filename):
            filename=secure_filename(file.filename)
            print(filename)
           
            
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            filenames1,filenames2,filenames3=test_modle_bingli.evaluate_one_image(UPLOAD_FOLDER+str(filename))
#             out_html = inference(file_path)
#             return redirect(url_for('upload_file')) 
#             str_ret2=str_ret2%filename
            return str_ret+'<br/><img src="./uploads/%s" wight=100 hight=100 border=1>'%filename+'<br/>This is a CLL with possibility'+str(filenames1)+'<br/><h7>This is a FL with possibility</h7>'+str(filenames2)+'<br/>This is a MCL with possibility'+str(filenames3)
#                 <img src="./uploads/{%filename%}" border=1>
    return str_ret

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
if __name__=='__main__':
	app.run('0.0.0.0',port=8080)
