# Epileptic Challenge
Aquest projecte aborda la detecció d'atacs epilèptics mitjançant l'anàlisi d'enregistraments d'electroencefalograma (EEG). Aquesta es una eina  no invasiva per explorar la funcionalitat del cervell en registrar l'activitat elèctrica dels neuronis durant les seves sinapsis. Aquesta activitat es registrara mitjançant un conjunt de 21 elèctrodes situats sobre el cuir cabellut *Figura 1*. Així, els enregistraments d'EEG proporcionen 23 senyals temporals 1D, que en el procés de diagnòstic de l'epilèpsia, el neuroleg visualitza de manera visual les senyals d'EEG enregistrades amb l'objectiu de trobar patrons de pics, ones agudes i ones lentes que caracteritzen una crisi epilèptica. Aquesta anàlisi és una tasca que requereix molt temps pels professionals de la salut, és per això que farem ús de dues xarxes neuronals (CNN i LSTM) per a la detecció i classificació de crisis epilèptiques en finestres de temps.

#### Figura 1
![1200px-21_electrodes_of_International_10-20_system_for_EEG svg](https://github.com/rauldaal/epileptic-challenge/assets/61145059/2788d529-a692-46b1-902a-9649601586bc)


*Exemple de posició d'eltrodes en EGG*

## Codi
L'estructura del projecte es la següent:
1. ``main.py``: Conté el codi principal del projecte, a l'executar-lo es posa en funcionament tot el sistema per entrenar/testejar els diferents models.
2. ``config.json``: Conte la configuració utilitzada durant el projecte.
3. ``handlers``:
   - ``__init__.py``: Imports llibreries.
   - ``cofiguration.py``: Carrega configuració per els paràmatres del model i permet multi-execució.
   - ``data.py``: Crida per recuperar les dades del dataset a través de la classe Dataset i crear els DataLoaders.
   - ``generator.py``: Genera els objectes model i les seves funcions derivades per guardar-lo i carregar-lo.
   - ``Kfold.py``:  Funcions per realitzar el k_fold i group_k_fold des d'on es fara l'entrenament dels models.
   - ``train.py``: Entrenament del model.
   - ``test.py``: Test i mètriques model.
4. ``objects``:
   - ``__init__.py``: Imports llibreries.
   - ``dataset.py``: Defineix classes dataset per carregar i guardar les dades.
   - ``model.py``: Defineix l'arquitectura dels models.
5. ``models``: Contenidor per guardar els models generats en format .pickle.
6. ``plots``: Contenidor per guardar les figures referents a les mètriques del model.

# Dataset

Hem treballat en un conjunt de dades procedents de *CHB-MIT Scalp EEG Database* on hem trebellat amb dades d'entre 5 i 7 pacients on cada pacient esta identificat amb ch01 - ch05. De cada pacient tenim les seves dades ``Annoted`` amb els corresponets parquet (metadades) i npz (EGG en format numpy)

En les metadades .parquet hi torbem la seguent ifnormació:

Per cada pacient, tenim un fila per finestra del EGG, on hi trobem la classe (Si hi ha hagut un atac epilèptic en aquell interval) i el filname que referencia al pacient.

I per les npz, un  archiu numpy corresponent a cada finestra per cada pacient

# Dataloader 

Per tant, pel desenvolupament del projecte s'ha definit una classe pare ``EpilepticDataset`` el qual agafara el dataset mencionat anteriorment. Del total d'aquestes dades el 80% s'utilitzaran per entrenament i el 20% restant per validació.

Especificar que el dataset ha estat creat de forma per crear un model generalitzar per poder classificar qualsevol pacient, es a dir hem dividit les dades de forma "patient-cross" on les finestres dels pacients de test no tenen finestres seves en l'entrenament i no distigeix segons tipus de pacient com ara edat etc.. per crear un model personalitzat per aquell grup.

A traves d'aquesta classe dataset es creara el dataloader.

# Metedologia i Arquitectures

La metodologia a seguir per classificar les finestres dels pacients segons si contenen o no un atac epilèpic consistira en fer us de dues xarxes neuronals diferents. En aquestes els hi arrivara la informació de les finestres de cada pacient i si conte o no un atac epilèptic, pero aquesta informació arriva de 21 canals diferents i es informació temporal, per tant s'han definit dues formes d'enfocar el problema.

A traves de una CNN on es fusionaran els canals d'entrada a tarves de l'arquitectura del model, utilitzant una convolució de 21 a 5 canals, un MaxPool de kernel 4 i un flatten per aplanar la sortida. 

Per poder abordar el problenma per poder aprofitar la informació temporal en el procees d'apranantatge hem definit una LSTM.

Per tant a traves d'aquestes xarxes neuronals que s'utilitzaran per l'entrenament podem definirla Pipeline que seguira el projecte *Figura 2*

#### Figura 2
![tempsnip](https://github.com/rauldaal/epileptic-challenge/assets/61145059/b5c0e4f5-9011-44ff-beda-a9233333a603)

*Challenge Pipeline*
*Debora Gil, Guillermo Torres, José Elias Yauri, Carles Sánchez*

### CNN - Arquitectura

La xarxa neuronal consta d'una capa de convolució 1D amb 21 canals d'entrada, 5 canals de sortida i un kernel de mida 6. A continuació, apliquem una capa de MaxPooling 1D amb un kernel de mida 4, un pas de 1 i un rebliment de 1 per reduir la dimensionalitat de les dades. Seguidament, la sortida d'aquesta capa passa a través d'una capa d'aplanat per convertir-la en un vector unidimensional.

Després, es segueixen dues capes totalment connectades: la primera amb 610 unitats de sortida i la segona amb 2 unitats de sortida. Aquestes capes transformen la representació de les dades. Per prevenir el overfitting, introduim un dropout amb una probabilitat del 0.15, que apaga aleatòriament algunes neurones durant l'entrenament.

Finalment, fem una capa completament connectada amb 2 unitats d'entrada i 1 unitat de sortida, seguida d'una capa d'activació com ara una sigmoide.

### LSTM - Arquitectura

La LSTM constara de 21 features d'entrada, 256 de hidden layer que es la que guarda la informació temporal i contindra 1 sola capa.

L'arquitectura de la capa squencial sera: La primera capa lineal transforma les dades d'entrada, que tenen un nombre de dimensions igual a hidden_size (la sortida d'una capa anterior), en un nou conjunt de dades amb hd unitats de sortida. Després d'aquesta transformació lineal, apliquem la funció d'activació ReLU, introduint no-linearitats.

La segona capa lineal pren les dades de sortida de la primera capa (amb hd unitats) i les transforma en un conjunt final amb n_classes=1 unitats de sortida. Aquesta capa lineal representa l'última transformació de la xarxa abans de la sortida. Finalment, a la sortida, apliquem una funció d'activació com ara una sigmoide.


## Entrenament model

Per l'entrenament del model s'han emprat els següents paràmetres , ``5 epoques`` amb la funció d'optimització ``Adam`` i un leraning rate de ``0.0001`` i un btach size de 600 ja que es el que recomana el model LSTM. 

Hem fet la implementació de dos tipus de K-Fold per l'entrenament, els quals s'han utilitzat en els dos tipus d'aqruitectura mencionats, per tant hem extret un total de 4 models diferents del que posteriorment veurem els seus resultats:

### K-Fold
L'objectiu d'utilitzar la tècnica de K-Fold volem separar les dades d'entrenament a dades d'entrenament i validació. D'aquesta forma els models veuran més dades i obtindran millors resultats, amb perillositat de sobre ajustar els models.

Fent servir un K-Fold per a totes les finestres de tots els pacients al conjunt d'entrenament una vegada separat el conjunt de test. Independentment del balanceig entre pacients aquest K-Fold no té en compte la distribució de les dades en generar els folds, per tant, pot ser que en separar les dades es generin folds sense dades d'alguns pacients.

### Grup K-Fold
Group K-Fold soluciona aquest problema de la distribució de pacients en els diferents folds. D'aquesta forma assegurem que per cada fold que es generi hi ha dades de tots els pacients, tant pels folds d'entrenament com els de validació.

## Mètriques i resultats

### Models CNN
|   | CNN + KFOLD | CNN + Grup KFOLD|
|---| -------------| ------------- | 
|Loss|![W B Chart 20_1_2024, 12_03_35](https://github.com/rauldaal/epileptic-challenge/assets/61145059/223ed8b3-c0f0-4098-9dae-cc52eb08eb68)|![W B Chart 20_1_2024, 12_08_29](https://github.com/rauldaal/epileptic-challenge/assets/61145059/a8ced896-b34c-4ccb-a142-035ade2b54c8)|

*Train i test loss*

Podem veure com los curves de la loss segueixen una curvatura semblant tant en train com en validation i aconseguint uns bon valors baixos, per tant descartem l'opcio de que estiguem tenint overfitting. Per altre banda veiem aquest patro en els tant en el model que utilitza Kfold, com en el GroupKfold.

#### Confusion Matrix
Observant la matriu de confusió, es veu com aconseguim un ``Accuracy X%``,


| CNN + KFOLD | CNN + Grup KFOLD |
| -------------| ------------- | 
|![image](https://github.com/rauldaal/epileptic-challenge/assets/88167948/31c7184a-e2c1-42f1-9d0a-5d04b8e45ab9) |![image](https://github.com/rauldaal/epileptic-challenge/assets/88167948/d45dca3c-a311-4728-be79-c0611c974a0f)| 

*Confusion Matrix*

Observant la matriu de confusió veiem en el cas de Kfold un accuracy del ``95%`` i un recall de ``97%``, per altre banda en el GroupKfold veiem un accuracy del ``97%`` i recall de ``98%``. Per tant per el moment el GroupKfold es el millor model que hem aconseguit.

### Models LSTM
|   | LSTM + KFOLD | LSTM + Grup KFOLD|
|---| -------------| ------------- | 
|Loss|![W B Chart 18_1_2024, 09_28_13](https://github.com/rauldaal/epileptic-challenge/assets/61145059/d5fa04a9-ed1c-40a2-b1de-0f178ce01fef)| ![W B Chart 18_1_2024, 11_09_25](https://github.com/rauldaal/epileptic-challenge/assets/61145059/7766e840-ce17-4719-bfb8-e1b0971e2206)|
|Test loss|![W B Chart 18_1_2024, 09_30_51](https://github.com/rauldaal/epileptic-challenge/assets/61145059/570a4e25-ade2-4fbb-9670-c81d45671ede)| ![W B Chart 18_1_2024, 12_25_14](https://github.com/rauldaal/epileptic-challenge/assets/61145059/cc83d511-b9b6-4db6-a206-63a511d3524e)|

*Train i test loss*

Podem veure com los curves de la loss segueixen una curvatura semblant tant en train com en validation, a diferencia de la CNN podem observar pics en la loss, pero aconseguint uns valors baixos, per tant descartem l'opcio de que estiguem tenint overfitting. Per altre banda veiem aquest patro en els tant en el model que utilitza Kfold, com en el GroupKfold.

#### Confusion Matrix
Observant la matriu de confusió, es veu com aconseguim un ``Accuracy X%``,


| LSTM + KFOLD |LSTM + Grup KFOLD|
| -------------| ------------- | 
|![plotsLSTM](https://github.com/rauldaal/epileptic-challenge/assets/61145059/66601c10-54c5-4d1d-bf16-88e92fdd618c)|![plotsgroupLSTM](https://github.com/rauldaal/epileptic-challenge/assets/61145059/80a274fe-dc10-4f36-875e-4b483f67a8e7)|

*Confusion Matrix*
Observant la matriu de confusió veiem en el cas de Kfold un accuracy del ``94%`` i un recall de ``96%``, per altre banda en el GroupKfold veiem un accuracy del ``93%`` i recall de ``95%``. Per tant per el moment el GroupKfold es el millor model que hem aconseguit.

## Conclusions
En base els resultats presentats, podem extreure les següents conclusions:

Els dos tipus de models (CNN i LSTM) presenten bons resultats en termes d'accuracy i recall, indicant que són capaços de fer una classificació efectiva en el conjunt de dades.

L'estratègia de validació creuada Group K-Fold tendeix a proporcionar lleugerament millors resultats en comparació de K-Fold per a ambdós tipus de models.

En general, els models CNN mostren un rendiment lleugerament superior en comparació dels models LSTM en aquest conjunt de dades específic.


## Contributors
* Sergi Tordera - 1608676@uab.cat
* Eric Alcaraz - 1603504@uab.cat                
* Raul Dalgamoni - 1599225@uab.cat





