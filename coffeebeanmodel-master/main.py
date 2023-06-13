import uvicorn
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = FastAPI()  # create a new FastAPI app instance

# port = int(os.getenv("PORT"))
port = 8080

model1 = tf.keras.models.load_model('lite_model.h5')
model2 = tf.keras.models.load_model('Kmeans.h5')

def houseDetection(file):
    img = image.load_img(file, target_size=(100,100,3))

    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model1.predict(images, batch_size=10)
    print(classes[0])

    if classes[0] > 0.3:
        output1 = 1
        print("User layak menerima bantuan")
    else:
        output1 = 0
        print("User tidak layak menerima bantuan")
    return output1

def salaryPrediction(pendapatan)
    data = pd.read_csv('combined_data.csv')

    nama = data['Nama'].values
    penghasilan = data['Penghasilan'].values
    cluster = data['Cluster'].values

    new_data = np.array([pendapatan])
    data_normalized = (new_data - penghasilan.min()) / (penghasilan.max() - penghasilan.min())
    data_normalized = data_normalized.reshape(-1, 1)
    cluster_probs = model2.predict(data_normalized)
    clusters = np.argmax(cluster_probs, axis=1)

    if clusters == 2:
        output2 = 1
        print("Kategori penghasilan user: rendah")
    elif:
        output2 = 0
        print("kategori penghasilan user: menengah")
    else:
        output2 = 0
        print("kategori penghasilan user: tinggi")
    return output2


@app.get("/")
def hello_world():
    return ("hello world")

@app.post("/3/")
def classifybean(input: UploadFile = File(...)):
    print(input.filename)
    print(type(input.filename))
    savefile = input.filename
    with open(savefile, "wb") as buffer:
        shutil.copyfileobj(input.file, buffer)
    result = somethingidk2(savefile)
    os.remove(savefile)
    return {result}
    
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)