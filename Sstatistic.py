from pandas import DataFrame, concat
from numpy import sqrt, random

class OutliersTest:
    def __init__(self, dataset, header):
        self.dataset = dataset
        self.header = header

class FrameOutliers(OutliersTest):
    def __init__(self, dataset, header):
        super().__init__(dataset, header)

    #crear la salida con la clase hijo
    def DatasetOutliers(self):
        self.Outliers = DataFrame()
        for i in self.header:
            #definir cuantiles
            Q1 = self.dataset[i].quantile(0.25)
            Q3 = self.dataset[i].quantile(0.75)
            
            #definir Limites
            IQR = Q3-Q1
            lower_lim = Q1-1.5 * IQR
            upper_lim = Q3+1.5 * IQR    
            print (f'{i}: Lower_lim: {lower_lim}, Upper_lim: {upper_lim}')        

            # tabla de Outlaiers
            Outliers_15_low = (self.dataset[i] < lower_lim)
            Outliers_15_up = (self.dataset[i] > upper_lim)
            TableOutlier = self.dataset[i][(Outliers_15_low|Outliers_15_up)]
            TableOutlier = DataFrame(TableOutlier)
            self.Outliers = concat([self.Outliers, TableOutlier])

        return self.Outliers

    def Peso(self):
        for i in self.Outliers:
            a = sum(self.Outliers[i].value_counts(dropna=True))
            a+=a

        print(f'el porcentaje de Outliers es: {round(a/(len(self.dataset)*len(self.dataset.columns))*100,1)}%')


class JarqueBera:
    def __init__(self, x, nrepl = 2001):
        self.x = x
        self.nrepl = nrepl

    def jb_norm_test(self):

        l = 0 #contador
        n = len(self.x.columns) #numero de columnas
        self.x1 = self.x.values.sum()/n
        b1 = sqrt(n)*((self.x-self.x1)**3).values.sum()/(((self.x-self.x1)**2).values.sum())**(3/2)
        b2 = n*((self.x-self.x1)**4).values.sum()/(((self.x-self.x1)**2).values.sum())**2
        t = (n/6)*((b1)**2 + ((b2-3)**2)/4)
                    
        for i in list(range(1,self.nrepl)):
            z = random.normal(size = n)
            z1 = z.sum()/n
            a1 = sqrt(n)*((z-z1)**3).sum()/(((z-z1)**2).sum())**(3/2)
            a2 = n*((z-z1)**4).sum()/(((z-z1)**2).sum())**2
            T = (n/6)*((a1)**2 + ((a2-3)**2)/4)
                
            if T>t: 
                l=l+1
                    
            pvalue = l/self.nrepl
        
        print('Jarque-Bera, Ruslan Pusev version 1.1'.center(50,'-'))
        print(f'Statistics: {round(t, 4)}\np-value: {round(pvalue, 4)}')