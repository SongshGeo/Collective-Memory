# Flooding Memory
This project aims to improve socio-hydrological models that simulate social and flood interactions using the Universal Decay Model proposed by Candia et al. The paper is in preparation and the full project code and data will be made public after the official publication.

## Research framework

 Assuming social memory of floods as an important partial, Di Baldassarre et al. (2013), Viglione et al. (2014), Di Baldassarre etal. (2015) raised models to capture the co-evolution of society and flood events in a socio-hydrological approach. On the base of their conceptualizations, we proposed a possible substitution of society module by the Universal Decay Model (UDM) of collective memory (Candia et al., 2019) as it provides a more explicit explanation to this kind of social process.

As all modules and their relations illustrated in **Fig. 1**, we choose the model by Di Baldassarre (2015) as a basic version to test the substitution, where collective memory plays a controlling role but at a lack of data as a support. First of all, we select a typical floodplain to do a survey regarding memory of major historical floods. Then, after processing the questionnaires data from survey, we use the dataset to fit memory decay rate under different alternatives of the society module. Finally, once this substitution works, we do simulations in socio-hydrological models to test if any essential difference can be demonstrated, otherwise there is no need to make the module more complicated.

## Document Index：

`generate_floods.py` provides the algorithms for generating random flood sequences.

`collective_memory.py` is the algorithm used in the simulation of the socio-hydrological model

`decay_rates_fit.py` provides a process for fitting memory decay rates using survey data.

`processing_questionnaires_data.ipynb` used in processing data.

`data/Processed_data_in_English.csv` is our survey data after processing.

## Versions and Dependencies

We use **python 3.7** and it works well.

Dependent libraries include: **Pandas**, **Numpy**

## Major references：

>**Di Baldassarre, Giuliano, et al. 
>"Debates—Perspectives on socio‐hydrology: 
>Capturing feedbacks between physical and social processes." 
>Water Resources Research 51.6 (2015): 4770-4781.**
>
>**Candia, C., C. Jara-Figueroa, C. Rodriguez-Sickert, A.-L. Barabási, and C. A. Hidalgo. 2019. The universal decay of collective memory and attention. *Nature Human Behaviour* 3(1):82–91.**
>
>**Ciullo, A., Viglione, A., Castellarin, A., Crisci, M., & Di Baldassarre, G. (2017). 
>Socio-hydrological modelling of flood-risk dynamics: comparing the resilience of green and technological systems. 
>Hydrological sciences journal, 62(6), 880-891.**
>
>**Viglione, Alberto, et al. 
>"Insights from socio-hydrology modelling on dealing with flood risk–roles of collective memory, risk-taking attitude and trust." 
>Journal of Hydrology 518 (2014): 71-82.**