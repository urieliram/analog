# Multiple analog forecasting
## A load forecasting method based on correlation and regression models.

The code of the models proposed are in [analogTS.ipynb](analogTS.ipynb). From there, the the function `analogTS` is highlighted:

```python
def analogTS(serie, vsele, k = 10, tol = 0.8, typedist = 'pearson', typereg = 'OLSstep', verbose = False, 
             # [ ] Agregar params que incluya lo de abajo
             n_components = 3,
             n_jobs = None):
    """_summary_

    Args:
        serie (_type_): _description_
        vsele (_type_): Size of the selection window / Tamanio de la ventana de selección
        k (int, optional): Number of neighbours to search for k / Número de vecinos a buscar k. Defaults to 10.
        tol (float, optional): Window size tolerance for neighbour selection / Tolerancia de tamaño de ventanas para seleccion de vecinos. Defaults to 0.8.
        n_components (int, optional): _description_. Defaults to 3.
        typedist (str, optional): distance measure, 'euclidian' or 'pearson' or 'dtw' / medida de distancia, 'euclidian' o 'pearson' o 'dtw' . Defaults to 'pearson'.
        typereg (str, optional): _description_. Defaults to 'OLSstep'.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    # STEP 1: Selection of the windows with the highest correlation.
    
    ## Start time mesurement
    t0 = time.time()
    n = len(serie) 
    
    ## Calculate the distance between all neighbors.
    distances = []
    Y = serie[n - vsele: n]           ## latest data
    for i in range(n - 2 * vsele):
        if  typedist == 'spearman':     ## dynamic time warping
            dist = np.corrcoef(Y, serie[i:i+vsele], method='spearman')[1,0]
        elif typedist == 'euclidian':
            dist = euclidean(Y, serie[i:i+vsele])
        else:
            dist = np.corrcoef(Y, serie[i:i+vsele])[1,0]
        if dist > 0:
            distances.append((i, dist))
        
    ## We calculate the neighbourhood by distance from smallest to largest and the positions are saved.
    if typedist == 'pearson' or typedist == 'spearman':
        ## In the Pearson backwards case, we are interested in the indices with the highest correlation in Pearson backwards ordering.
        distances.sort(key=lambda tup: tup[1], reverse=True)
    else:
        distances.sort(key=lambda tup: tup[1], reverse=False)

    positions  = []
    neighbors  = [] # X, o vecinos
    neighbors2 = [] # X', o consecutivos de los vecinos

    ## We calculate the k nearest neighbors and save the positions.
    i = 0
    for pos, _ in distances:
        if i == 0:      
            positions.append(pos)   
            neighbors.append(serie[pos:pos+vsele])
            neighbors2.append(serie[pos+vsele:pos+2*vsele])  
        else:
            bandera = True
            for p in positions:
                 ## if we already had a position in the list that passed the tolerance, we no longer save it
                # [ ] Para evitar repetir vecinos que estén dentro de cierta tolerancia
                if (abs(pos - p) < tol * vsele):
                    bandera = False
                    i = i - 1
                    break
            if bandera == True:
                ## save new neighbor
                positions.append(pos)   
                neighbors.append(serie[pos:pos+vsele])
                neighbors2.append(serie[pos+vsele:pos+2*vsele])  
                bandera = False
        i = i + 1
        if i == k: # When the number of neighbors is reached
            break
    
    neighbors  = np.array(neighbors)
    neighbors2 = np.array(neighbors2)
    
    if verbose == True:
        print('positions KNN:', positions) ## position of k nearest neighbors
        
        # Image
        fig, ax1 = plt.subplots(figsize=(8, 5))
        plt.title('Selección con KNN:',fontsize = 'x-large',color = '#ff8000')
        ax1.set_xlabel('Tiempo', color = '#ff8000', fontsize = 'large')
        ax1.set_ylabel('Demanda', color = '#ff8000', fontsize = 'large')
        plt.tick_params(colors = '#ff8000', which='both')
        ax1.spines['bottom'].set_color('#ff8000')
        ax1.spines['top'].set_color('#ff8000') 
        ax1.spines['right'].set_color('#ff8000')
        ax1.spines['left'].set_color('#ff8000')
        if len(Y) != 0: 
            plt.plot(Y,alpha=0.6, linestyle='dashed', color='red', linewidth=3)
        for p in neighbors:
            plt.plot(p,alpha=0.3, linewidth=2)    
        plt.savefig(f'results/imgs/neighbors_{typedist}_{typereg}', transparent = True)         
        plt.show()

    ## End time mesurement
    t_sel = time.time() - t0

    # STEP 2: Regression between nearest neighbors 'X' and last window 'Y'

    # Recalculate start time
    t0 = time.time()
    ## Define our regressors
    X   = (neighbors.T ).tolist()
    X_2 = (neighbors2.T).tolist()
    Y   = (Y).tolist()
    prediction_Y2 = []
    
    prediction_Y2 = regression_fit(X, Y, typereg, n_jobs = n_jobs)

    if verbose==True:
        # Image
        fig, ax1 = plt.subplots(figsize=(8, 5))
        plt.title('Selección con KNN:',fontsize = 'x-large',color = '#ff8000')
        ax1.set_xlabel('Tiempo', color = '#ff8000', fontsize = 'large')
        ax1.set_ylabel('Demanda', color = '#ff8000', fontsize = 'large')
        plt.tick_params(colors = '#ff8000', which='both')
        ax1.spines['bottom'].set_color('#ff8000')
        ax1.spines['top'].set_color('#ff8000') 
        ax1.spines['right'].set_color('#ff8000')
        ax1.spines['left'].set_color('#ff8000')
        if len(prediction_Y2) != 0: 
            plt.plot(prediction_Y2,alpha=0.6, linestyle='dashed', color='red', linewidth=3)
        for p in neighbors2:
            plt.plot(p,alpha=0.3, linewidth=2)    
        plt.savefig(f'results/imgs/neighbors2_{typedist}_{typereg}', transparent = True)
        plt.show()

    t_reg = time.time() - t0
    fail_ = False
    if len(prediction_Y2) == 0:
        prediction_Y2=[serie[-1]] * vsele
        fail_=True
        print(">>> analogo_knn: Forecast not calculated.")

    return prediction_Y2, t_sel, t_reg, fail_, positions
```

Also, the Jupyter Notebook contains a cell with the code of the metafunction described in the paper:

```python
timeAn_         = []
forecastAn_     = []
forecastAnMA_   = []
forecastX_An_   = []
forecastX_AnMA_ = []
row             = []
rowX            = []
row2            = []
rowX2           = []
k               = 5          ## Number of nearest neighbors
tol             = 0.8        ## Closest tolerance percentage between neighbors
typedist        = 'pearson'  ## Distance betweeen neighbors: 'pearson' 'euclidian' 'lb_keogh' 'matrixprofile'
typereg         = 'PCR'  ## Regression model: 'OLSstep' 'Boosting' 'Bagging' 'LinearReg' 'AdaBoost' 'BayesRidge' 'LassoReg' 'RidgeReg' 'PLS' 'PCR' 'VotingEnsemble' 'VotingLinear'
n_p             = n_p        ## Number of periods per step
nfail           = 0
fail_           = False
vsele           = vsele      ## Number of periods in a window

for to,tt,tf in positions_test:
    j=0
    s=n_p
    for i in range(tt,tf,1):
        ## Analogue method parameters
        X_train = np.array(serie1[to+j:tt+j])
        t_o = time.time()
        try:
            pred_, t_sel_, t_reg_, fail_, positions = analogTS(X_train, vsele=vsele, k=k, tol=tol, typedist=typedist, typereg=typereg, verbose=False)
        except:
            print("!!! Error has occurred in the position:",tt+j)
            # row = ['!!! Error has occurred in the position:',tt+j]
            # append_list_as_row('LogAn.csv',row)
        if fail_==True:              
            nfail = nfail + 1
            print(">>> Persistence forecast in position:",tt+j)
        pred_list  = pred_.tolist()
        forecastAn_ = forecastAn_ + pred_list[0:1]
        row = [tt+j] + [time.time() - t_o] + pred_list[0:1] + [t_sel_] + [t_reg_]
        append_list_as_row('LogAn_PCR.csv',row)
        print(j+tt)
        # Minimum error in the first forecast
        if j >= 3 * n_p:
            a = - 1 - min(j,vsele) 
            b = - 1
            array1 = np.array(forecastAn_[a:b])
            c = tt + j - min(j,vsele)
            d = tt + j
            array2 = np.array(serie1[c:d])                         
            epsilon = np.subtract(array1,array2)  ## Errores del primer pronóstico
            ar = AutoReg(epsilon, lags=int(n_p/2)).fit() ## Ajuste de los últimos n_p/2          
            delta = ar.forecast(1)
            MA_ = pred_list[0:1] - delta
            forecastAnMA_.extend(MA_)
            MA_ = [MA_]
            row = [tt+j] + [time.time() - t_o] + MA_ + [t_sel_] + [t_reg_]
            append_list_as_row('LogAnMA_PCR.csv',row)
            if s==n_p:
                forecastX_An_   = forecastX_An_ + pred_list[0:n_p]
                for val in pred_list[0:n_p]:
                    row = [tt+j] + [time.time() - t_o] + [val]
                    append_list_as_row('LogXAn_PCR.csv',row)

                MA_ = pred_list[0:n_p] - delta
                forecastX_AnMA_.extend(MA_)
                for val in MA_[0:n_p]:
                    row = [tt+j] + [time.time() - t_o] + [val]
                    append_list_as_row('LogXAnMA_PCR.csv',row)
                s=0
        else:
            if s==n_p:
                forecastX_An_   = forecastX_An_ + pred_list[0:n_p]
                forecastX_AnMA_ = forecastX_An_
                for val in pred_list[0:n_p]:
                    row = [tt+j] + [time.time() - t_o] + [val]
                    append_list_as_row('LogXAn_PCR.csv',row)
                    append_list_as_row('LogXAnMA_PCR.csv',row)
                s=0
            forecastAnMA_ = forecastAn_
            row = [tt+j] + [time.time() - t_o] + [forecastAn_[-1]] + [t_sel_] + [t_reg_]
            append_list_as_row('LogAnMA_PCR.csv',row)

        timeAn_.append(time.time() - t_o)
        j = j + 1
        s = s + 1

    modu=(tf-tt)%n_p
    if modu != 0:
        end=min(len(forecastAn_),len(forecastX_An_))
        print('modu',modu)
        forecastAn_     = forecastAn_[    0:end]
        forecastAnMA_   = forecastAnMA_[  0:end]
        forecastX_AnMA_ = forecastX_AnMA_[0:end]        
        forecastX_An_   = forecastX_An_[  0:end]
print('>>> Number of forecasts not calculated:', nfail)
```