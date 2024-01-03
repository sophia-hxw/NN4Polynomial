import json

def writejson(file_name = 'test', data={'hello': 'world'}):
    # 将数据写入 JSON 文件
    with open(file_name + '.json', 'w') as json_file:
        json.dump(data, json_file)

    print("Done JSON file written.")

def get_params(file_name = './configs/params_fcn.json'):
    with open(file_name, 'r') as file:
        params = json.load(file)
    
    return params['data'], params['training'], params['model'], params['testing']

def vis_table(test_dataset, pred_y, test_loss):
    # 设置列宽度
    print_width = 10

    # 打印表头
    print("{:<{}} {:<{}} {:<{}} {:<{}}".format("test_x", print_width, "test_y", print_width, "pred_y", print_width, "MSELoss", print_width))
    print("-" * (print_width  + print_width  + print_width + print_width))

    # 打印数据
    for idx in range(len(test_dataset)):
        print("{:<{}} {:<{}} {:<{}} {:<{}}".format(round(test_dataset.x[idx].item(),4), print_width, round(test_dataset.y[idx].item(),4),  print_width, round(pred_y[idx],4),  print_width, round(test_loss[idx],4), print_width))

def save_json(file_name, test_dataset, pred_y, test_loss):
    datas = []

    for idx in range(len(test_dataset)):
        oneres = {}
        oneres["test_x"] = round(test_dataset.x[idx].item(),4)
        oneres["test_y"] = round(test_dataset.y[idx].item(),4)
        oneres["pred_y"] = round(pred_y[idx],4)
        oneres["MSELoss"] = round(test_loss[idx],4)
        datas.append(oneres)

    wjson = {}
    wjson["results"] = datas
    writejson(file_name, wjson)
