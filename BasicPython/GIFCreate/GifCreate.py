# from 相约机器人
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

file_path = '/Users/xj2sgh/PycharmProjects/OneHourProgram/BasicPython/overdose_data_1999-2015.xls'
overdose = pd.read_excel(file_path, sheeaname='Online', skiprows=6)
def get_data(table, rownum, title):
    data = pd.DataFrame(table.loc[rownum][2:]).astype(float)
    data.columns = {title}
    return data

def animate(i):
    data = overdose.iloc[:int(i+1)]
    p = sns.lineplot(x=data.index, y=data[title], data=data, color='r')
    p.tick_params(labelsize=7)
    plt.setp(p.lines, linewidth=7)

def augment(xold, yold, numsteps):
    xnew = []
    ynew = []
    for i in range(len(xold)-1):
        difx = xold[i+1]-xold[i]
        stepsx = difx/numsteps
        dify = yold[i+1]-yold[i]
        stepsy = dify/numsteps
        for s in range(numsteps):
            xnew = np.append(xnew, xold[i]+stepsx)
            ynew = np.append(ynew, yold[i]+stepsy)
    return xnew,ynew

def smoothListGaussian(listin, strippedXs=False, degree=5):
    window = degree*2-1
    weight = np.array([1.0]*window)
    weightGauss = []
    for i in range(window):
        i = i-degree+1
        frac = i/float(window)
        gauss = 1/(np.exp((4*(frac))**2))
        weightGauss.append(gauss)
    weight = np.array(weightGauss)*weight
    smoothed = [0.0]*(len(listin)-window)
    for i in range(len(smoothed)):
        smoothed[i] = sum(np.array(listin[i:i+window])*weight)/sum(weight)

    return smoothed

if __name__ == '__main__':
    title = 'Heroin Overdoses'
    d = get_data(overdose,18,title)
    x = np.array(d.index)
    y = np.array(d['Heroin Overdoses'])
    # x,y = augment(x,y,10)
    overdose = pd.DataFrame(y,x)
    overdose.columns = {title}
    writer = animation.writers['ffmpeg']
    writer = writer(fps=20, metadata=dict(artist='Me', bitrate=1800))
    fig = plt.figure(figsize=(10,12))
    plt.xlim(1999, 2016)
    plt.ylim(np.min(overdose)[0], np.max(overdose)[0])
    plt.xlabel('Year', fontsize=20)
    plt.ylabel(title, fontsize=20)
    plt.title(title+' '+'per Year', fontsize=20)

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=17, repeat=True)
    plt.show()