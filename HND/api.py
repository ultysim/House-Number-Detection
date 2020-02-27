import pickle
from sklearn.ensemble import RandomForestClassifier
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import numpy as np
import HND
import cv2

app = Flask(__name__)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('Image')


class PredictCancer(Resource):
    def get(self):
        # use parser and find the user's query

        model = HND.HouseNumberDetector()
        args = parser.parse_args()
        for i in parser.parse_args():
            print(i)
        user_query = args['Image']
        print(args)
        print(args['Image'])
        print(np.array(user_query).shape)

        pred = model.predict_numbers(user_query)
        print(pred)

        # create JSON object
        output = {'Numbers': pred}

        return output

api.add_resource(PredictCancer, '/')

if __name__ == '__main__':
    app.run(port=3000, debug=True)