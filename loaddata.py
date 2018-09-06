import numpy as np

questionpath = 'dataset/dataset-question1.txt'
answerpath = 'dataset/dataset-answer1.txt'


def load(questionpath, answerpathans):
    qf = open(questionpath, 'r')

    questiontext = qf.read()
    questionline = questiontext.split("\n")

    input_data = []
    output_data = []

    for line in questionline[:-1]:
        dic = eval(line)
        id = dic['id']
        # processing_times = np.array(dic['M'])
        numberofmashine = dic['m']
        numberofjob = dic['n']
        processing_times = np.zeros((numberofmashine, numberofjob))
        for ci in range(numberofmashine):
            for cj in range(numberofjob):
                processing_times[ci, cj] = dic['M'][ci][cj]

        sumtimeforeachjob = processing_times.sum(0)
        sumtimeforeachmachine = processing_times.sum(1)
        sumprosessingtime = [processing_times.sum().sum()]
        avetime_machine = sumtimeforeachmachine.sum()/numberofmashine
        avetime_job = sumtimeforeachjob.sum()/numberofjob

        for i in range(numberofjob):
            data1 = [
                id,                                     # id
                1/numberofjob,                          # number of job
                1/numberofmashine,                      # number of machine
                sumtimeforeachjob[i]/avetime_job,
                np.mean(processing_times[:, i]),         # average
                np.std(processing_times[:, i]),          # std
                np.min(processing_times[:, i]),          # min
                np.max(processing_times[:, i]),          # max
                np.median(processing_times[:, i]),          # max
            ]
            data2 = [
                float(processing_times[j, i]) for j in range(numberofmashine)
            ]
            data3 = [
                float(processing_times[j, i] / sumprosessingtime) for j in range(numberofmashine)
            ]
            data4 = [
                float(processing_times[j, i] / sumtimeforeachmachine[j]) for j in range(numberofmashine)
            ]
            dataitem = data1+data2+data3+data4
            input_data.append(dataitem)

    af = open(answerpath, 'r')
    answertext = af.read()
    answerline = answertext.split("\n")
    line1 = [i+1 for i in range(numberofjob)]
    for line in answerline[:-1]:
        dic = eval(line)
        for i in range(numberofjob):
            out = line1.index(dic['order'][i])
            output_data.append(out)
        # print(dic)
    return input_data, output_data


def main():

    questionpath = 'dataset/dataset-question1.txt'
    answerpath = 'dataset/dataset-answer1.txt'
    input_data, output_data = load(questionpath, answerpath)

    print(input_data)
    print(output_data)


if __name__ == '__main__':
    main()
