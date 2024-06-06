from flask import Flask, request, redirect, url_for, render_template
import os
from code_verif.feature_extraction import make_feature
import scipy.io
import torch
import numpy as np
import json

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)


UPLOAD_FOLDER = 'D:/VILAB/AIproject/uploads'  # 절대 파일 경로
SAVE_FOLDER = 'D:/VILAB/AIproject/mat'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/add')
def add():
    return render_template('add.html')

@app.route('/imgadding_d')
def imgadding_d():
    return render_template('imgadding_d.html')

@app.route('/imgadding_r')
def imgadding_r():
    return render_template('imgadding_r.html')

@app.route('/simverif')
def simverif():
    return render_template('simverif.html')

@app.route('/simverif_upload', methods=['GET', 'POST'])
def simverif_upload():
    global UPLOAD_FILE
    if request.method == 'POST':
        file = request.files['savefile'] # HTML 폼에서 파일을 업로드할 때 사용되는 input 필드의 이름.
        if file and allowed_file(file.filename):
            filename = file.filename
            name = filename.split('\\')[-1]

            #filepathtosave = os.path.join(UPLOAD_FOLDER, filename)
            filepathtosave = os.path.join(UPLOAD_FOLDER, 'query.jpg')
            file.save(filepathtosave)
            #로컬에 저장 후 로컬 경로 참조
            feat = make_feature(os.path.join(UPLOAD_FOLDER, 'query.jpg'))

            # 이미지 검증 프로세스 <= 사전에 저장된 mat파일을 load해서 추출한 feature랑 비교해야함.
            query_feature = feat
            result = scipy.io.loadmat(os.path.join(SAVE_FOLDER, 'data_represent.mat'))#'gallery_with_train.mat'
            gallery_feature = torch.FloatTensor(result['feat'])
            query_feature = query_feature.cuda()
            gallery_feature = gallery_feature.cuda()

            index, score = sort_img(query_feature,gallery_feature) # 파일명과 점수를 보내야함. 인덱스는 매칭이 안될텐데
                        
            top10_name = result['name'][index[:5]]
            top10_score = score[:5]
            top10_name = [n.strip() for n in top10_name]
            top10_score = [s for s in top10_score]
            print('filename :',top10_name)
            print(top10_score)
            
            # 이미지 저장 프로세스 <= 이미지 저장 여부에 대한 기준이 필요함 (지금은 파일명의 중복을 따지는 걸로)
            #result = {'feature':feat.numpy(),'name':name} 이거 기존 저장 방식 참고해서 수정해야함.
            #scipy.io.savemat(os.path.join(SAVE_FOLDER, 'data.mat'),result)

            # demo 파일에서 retrieval process 따와야해
            # 이미지 주소, similarity score는 별개로 가져와야한다 => 가져와서 웹에 노출시키는 코드도 필요하고, html도 이미지 띄울 수 있게 구현해야하고
            return redirect(url_for("simverif_result", name =top10_name, score=top10_score)) 
            #return redirect(url_for("simverif_result", name = top10_name, score=top10_score)) 
        #render_template('simverif_result.html', top10_name = top10_name, top10_score=top10_score) 
    return redirect(url_for("simverif")) #render_template('simverif.html')


@app.route('/simverif_result', methods = ['GET', 'POST'])
def simverif_result():
    name = request.args.getlist('name')
    score = request.args.getlist('score')
    return render_template('simverif_result.html', rank_name=name, rank_score=score)

def sort_img(qf, gf):
    query = qf.view(-1,1)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]

    return index, score[index]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS



if __name__ == '__main__':
#    app.run(port=5001, debug=True)
    app.run(port=5001)

