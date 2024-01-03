from utils.visualize import readjson

if __name__ == "__main__":
    json_params = readjson()

    print(json_params['hidden_dims'])
    print(type(json_params['hidden_dims']))