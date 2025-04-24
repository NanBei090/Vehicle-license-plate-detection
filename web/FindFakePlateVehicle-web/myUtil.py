# 定义一个函数，根据车牌号查询比对车辆信息库（模拟车管所车辆登记信息库），判定是否是套牌车（车牌号和车辆品牌信息对应不上，后续可再增加车辆颜色信息）
def isFakePlate(inputCarInfo, carInfoDatabase):
    carBrandList = []
    isFakePlateCar = False
    trueCarBrand = ''
    plateNo = inputCarInfo[0]
    carBrand = inputCarInfo[1]
    if carBrand == '其他':
        isFakePlateCar = False
        trueCarBrand = 'Null'
    else:
        result = carInfoDatabase[(carInfoDatabase['plateNo'] == plateNo)]  # 从车管所数据库中拉出车牌号对应的车辆信息，保存到result中
        if len(result) > 0:
            carBrandList = result['carBrand'].values  # list结构 

            if carBrand == carBrandList[0]:
                # print(carBrand, "==", carBrandList[0])
                isFakePlateCar = False
                trueCarBrand = carBrandList[0]
            else:
                # print(carBrand, "!=", carBrandList[0])
                isFakePlateCar = True
                trueCarBrand = carBrandList[0]
        else:
            isFakePlateCar = False
            trueCarBrand = 'Null'
    return isFakePlateCar, trueCarBrand
