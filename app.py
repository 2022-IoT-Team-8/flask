from flask import Flask, request, jsonify
import pickle
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# model = pickle.load(open('PycharmProjects/app/test_save.pkl', 'rb'))
# model = DecisionTreeClassifier(random_state=34)
model = joblib.load('PycharmProjects/app/test_save.pkl')

app = Flask(__name__)

@app.route("/")
def index():
    return "flask in online"


@app.route('/predict', methods=['GET', 'POST'])
def location():
    data = request.get_json(force=True)
    prediction = model.predict([[np.array(
        data['94:64:24:9d:d8:52'],
        data['94:64:24:9d:d8:72'],
data['94:64:24:9d:d9:40'],
data['94:64:24:9d:d9:60'],
data['94:64:24:9d:de:e2'],
data['94:64:24:9d:df:02'],
data['94:64:24:9d:df:42'],
data['94:64:24:9d:df:62'],
data['94:64:24:9d:e6:f0'],
data['94:64:24:9d:e7:10'],
data['94:64:24:9d:f1:12'],
data['94:64:24:9d:f1:32'],
data['94:64:24:9d:f9:20'],
data['94:64:24:9d:f9:40'],
data['94:64:24:9d:f9:e0'],
data['94:64:24:9d:fa:00'],
data['94:64:24:9d:fb:f2'],
data['94:64:24:9d:fc:12'],
data['94:64:24:9e:03:e2'],
data['94:64:24:9e:16:02'],
data['94:64:24:9e:16:22'],
data['94:64:24:9e:21:92'],
data['94:64:24:9e:24:10'],
data['94:64:24:9e:24:30'],
data['94:64:24:9e:2a:d2'],
data['94:64:24:9e:2a:f2'],
data['94:64:24:9e:2c:e2'],
data['94:64:24:9e:2d:02'],
data['94:64:24:9e:37:50'],
data['94:64:24:9e:3e:e0'],
data['94:64:24:9e:3f:00'],
data['94:64:24:9e:4d:b0'],
data['94:64:24:9e:4d:d0'],
data['94:64:24:9e:54:42'],
data['94:64:24:9e:54:62'],
data['94:64:24:9e:61:30'],
data['94:64:24:9e:61:50'],
data['94:64:24:9e:72:42'],
data['94:64:24:9e:72:d2'],
data['94:64:24:9e:72:f2'],
data['94:64:24:9e:7c:f0'],
data['94:64:24:9e:7d:10'],
data['94:64:24:9e:7e:42'],
data['94:64:24:9e:7e:62'],
data['94:64:24:9e:7e:a0'],
data['94:64:24:9e:7e:c0'],
data['94:64:24:9e:7e:d2'],
data['94:64:24:9e:82:32'],
data['94:64:24:9e:82:42'],
data['94:64:24:9e:84:!2'],
data['94:64:24:9e:84:12'],
data['94:64:24:9e:84:32'],
data['94:64:24:9e:86:a0'],
data['94:64:24:9e:8a:72'],
data['94:64:24:9e:8a:92'],
data['94:64:24:9e:8a:d2'],
data['94:64:24:9e:8a:f2'],
data['94:64:24:9e:c0:a2'],
data['94:64:24:9e:c0:c2'],
data['94:64:24:9e:c3:12'],
data['94:64:24:9e:c3:32'],
data['94:64:24:9e:ce:a0'],
data['94:64:24:9e:d1:42'],
data['94:64:24:9e:e1:32'],
data['94:64:24:9e:fa:62'],
data['94:64:24:9e:fa:82'],
data['94:64:24:9f:03:c2'],
data['94:64:24:9f:03:e2'],
data['94:64:24:9f:14:c0'],
data['94:64:24:9f:2a:f2'],
data['94:64:24:9f:2b:12'],
data['94:64:24:9f:38:12'],
data['94:64:24:9f:38:32'],
data['94:64:24:9f:38:72'],
data['94:64:24:9f:38:92'],
data['94:64:24:9f:3c:f2'],
data['94:64:24:9f:3d:12'],
data['94:64:24:9f:51:62'],
data['94:64:24:9f:51:82'],
data['94:64:24:9f:70:72'],
data['94:64:24:9f:81:c2'],
data['94:64:24:9f:81:e2'],
data['94:64:24:9f:82:22'],
data['94:64:24:9f:82:42'],
data['94:64:24:9f:82:72'],
data['94:64:24:9f:83:72'],
data['94:64:24:9f:83:92'],
data['94:64:24:9f:9f:60'],
data['94:64:24:9f:9f:80'],
data['94:64:24:9f:a8:90'],
data['94:64:24:9f:a8:b0'],
data['94:64:24:9f:ae:c2'],
data['94:64:24:9f:ae:e2'],
data['94:64:24:9f:b2:20'],
data['94:64:24:9f:b2:40'],
data['94:64:24:9f:ba:92'],
data['94:64:24:9f:ba:b2'],
data['94:64:24:9f:c4:22'],
data['94:64:24:9f:c4:42'],
data['94:64:24:9f:ea:90'],
data['94:64:24:9f:ea:b0'],
data['94:64:24:9f:f9:62'],
data['94:64:24:9f:f9:82'],
data['94:64:24:9f:fb:12'],
data['94:64:24:a0:0b:82'],
data['94:64:24:a0:16:10'],
data['94:64:24:a0:16:30'],
data['94:64:24:a0:1f:70'],
data['94:64:24:a0:1f:90'],
data['94:64:24:a0:24:82'],
data['94:64:24:a0:24:a2'],
data['94:64:24:a0:34:d0'],
data['94:64:24:a0:34:f0'],
data['94:64:24:a0:3c:e2'],
data['94:64:24:a0:3d:02'],
data['94:64:24:a0:40:70'],
data['94:64:24:a0:40:90'],
data['94:64:24:a0:48:20'],
data['94:64:24:a0:48:40'],
data['94:64:24:a0:62:e0'],
data['94:64:24:a0:68:90'],
data['94:64:24:a0:68:b0'],
data['94:64:24:a0:70:60'],
data['94:64:24:a0:83:80'],
data['94:64:24:a0:89:f2'],
data['94:64:24:a0:8a:12'],
data['94:64:24:a0:8f:90'],
data['94:64:24:a0:8f:b0'],
data['94:64:24:a0:98:00'],
data['94:64:24:a0:ac:02'],
data['94:64:24:a0:ae:b2'],
data['94:64:24:a0:ae:d2'],
data['94:64:24:a0:cb:22'],
data['94:64:24:a0:cb:c0'],
data['94:64:24:a0:cb:e0'],
data['94:64:24:a0:ce:62'],
data['94:64:24:a0:ce:82'],
data['94:64:24:a0:cf:50'],
data['94:64:24:a0:cf:70'],
data['94:64:24:a0:d3:70'],
data['94:64:24:a0:d3:90'],
data['94:64:24:a0:d8:72'],
data['94:64:24:a0:df:a2'],
data['94:64:24:a0:df:c2'],
data['94:64:24:a0:e7:80'],
data['94:64:24:a0:e7:a0'],
data['94:64:24:a0:fe:30'],
data['94:64:24:a0:fe:50'],
data['94:64:24:a0:fe:c0'],
data['94:64:24:a0:fe:e0'],
data['94:64:24:a1:00:a2'],
data['94:64:24:a1:00:c2'],
data['94:64:24:a1:07:92'],
data['94:64:24:a1:07:b2'],
data['94:64:24:a1:08:b2'],
data['94:64:24:a1:08:d2'],
data['94:64:24:a1:09:72'],
data['94:64:24:a1:09:92'],
data['94:64:24:a1:0a:30'],
data['94:64:24:a1:0a:50'],
data['94:64:24:a1:0f:32'],
data['94:64:24:a1:22:80'],
data['94:64:24:a1:59:b2'],
data['94:64:24:a1:59:d2'],
data['94:64:24:a1:6f:72'],
data['94:64:24:a1:6f:92'],
data['94:64:24:a1:71:e2'],
data['94:64:24:a1:72:02'],
data['94:64:24:a1:7c:00'],
data['94:64:24:a1:7c:20'],
data['94:64:24:a1:88:f2'],
data['94:64:24:a1:89:12'],
data['94:64:24:a1:89:50'],
data['94:64:24:a1:89:70'],
data['94:64:24:a1:94:f2'],
data['94:64:24:a1:95:12'],
data['94:64:24:a1:99:d2'],
data['94:64:24:a1:99:f2']
    )]])
    output = prediction[0]
    return jsonify(output)

@app.route('/predict2', methods=['GET', 'POST'])
def position():
    data = request.get_json(force=True)
    #prediction = model.predict([[np.array(data['11:11:11'], data['12'],data['MAC3'])]])
    #output = prediction[0]
    return jsonify(data)


if __name__ == '__main__':
    app.run(host ='0.0.0.0', port = 5000, debug = True)

