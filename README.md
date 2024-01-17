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

# Metedologia i arquitectura

La metodologia a seguir per classificar les finestres dels pacients segons si contenen o no un atac epilèpic consistira en fer us de dues xarxes neuronals diferents. En aquestes els hi arrivara la informació de les finestres de cada pacient i si conte o no un atac epilèptic, pero aquesta informació arriva de 21 canals diferents i es informació temporal, per tant s'han definit dues formes d'enfocar el problema.

A traves de una CNN on es fusionaran els canals d'entrada a tarves de l'arquitectura del model, utilitzant una convolució de 21 a 5 canals, un MaxPool de kernel 4 i un flatten per aplanar la sortida. 

Per poder abordar el problenma per poder aprofitar la informació temporal en el procees d'apranantatge hem definit una LSTM.

Per tant a traves d'aquestes xarxes neuronals que s'utilitzaran per l'entrenament podem definirla Pipeline que seguira el projecte *Figura 2*

#### Figura 2
![tempsnip](https://github.com/rauldaal/epileptic-challenge/assets/61145059/b5c0e4f5-9011-44ff-beda-a9233333a603)
*Challenge Pipeline*
*Debora Gil, Guillermo Torres, José Elias Yauri, Carles Sánchez*

### CNN - Arquitectura

L'estrategia a seguir en la CNN donat que teniom 21 canals, sera fer ús 
