import pymongo
import pandas as pd
import pickle

models = ['classification_model_saved.sav', 'regression_model_saved.sav']
classification_model = pickle.load(open(models[0], 'rb')) # loading the model file from the storage
regression_model = pickle.load(open(models[1], 'rb')) # loading the model file from the storage

class Bulk_Predictor:
    def __init__(self, client, db, collection):
        print("inside constractor")
        self.client = str(client)
        print(client) 
        self.db = str(db)
        self.collection = str(collection)
        self.client = pymongo.MongoClient(self.client)
        print(self.client, "clienttttttt")
        self.db = self.client[self.db]
        self.collection = self.db[self.collection]
        

    def predictAndFetchRecord(self):
        print("getRecords")
        results = []
        df = pd.DataFrame(columns=['day', 'month', 'year', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'])
        for i in self.collection.find():           
            mydict = {'day': i['day'], 'month': i['month'], 'year': i['year'],
                        'RH': i['RH'], 'Ws': i['Ws'], 'Rain': i['Rain'],
                        'FFMC': i['FFMC'], 'DMC': i['DMC'], 'DC': i['DC'],   
                        'ISI': i['ISI'], 'BUI': i['BUI'], 'FWI': i['FWI'],                         
            }
            df.loc[-1] = mydict.values()
            df.index = df.index + 1  # shifting index
            df = df.sort_index()  # sorting by index
            results.append(mydict)
        def f_reg(RH,Ws,Rain,FFMC,DMC,DC,ISI):
                return regression_model.predict([[RH,Ws, Rain,FFMC,DMC,DC,ISI]])[0]
        def f_class(RH,Ws,Rain,FFMC,DMC,DC,ISI):
            if classification_model.predict([[RH,Ws, Rain,FFMC,DMC,DC,ISI]])[0] == 0:
                return "Not Fire"
            else:
                return "Fire"

        df['prediction temp'] = df.apply(lambda x: f_reg(x['RH'], x['Ws'], x['Rain'], x['FFMC'], x['DMC'], x['DC'], x['ISI']), axis = 1)
        df['prediction classes'] = df.apply(lambda x: f_class(x['RH'], x['Ws'], x['Rain'], x['FFMC'], x['DMC'], x['DC'], x['ISI']), axis = 1)

        return df