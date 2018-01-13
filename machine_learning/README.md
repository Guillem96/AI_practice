Decisicions de implementació de el test preformance i cross validation


A l'hora de fer l'arbre sobre el set d'entrenament hem decidit utilitzar crossvalidation, per fer-ho em optat per utilitzar una proporció que indicarà sobra quina part de les dades es montarà l'arbre i sobre quina es testejarà abans de testejar-ho en el test real. Hem decidit utilitzar una proporció i generar la part de les dades que utilitzarem per testejar sobre la resta del training set anant generant números aleatoris i anar distribuint el training set en les dades amb les que construirem l'arbre i les dades sobre el que el testejarem. Aquesta tasca la repetirem un nombre que pasarem per parametre de vegades i triarem l'arbre que més percentatje d'encerts ha aconseguit. Finalment aquest arbre el testejarem sobre la part que haviam deixat en un principi a part com a test i que no em utilitzat en cap moment a l'hora de construir l'arbre. El percentatje d'encerts que ens surti serà el que donarem però en el cas de que tinguesam de construir l'arbre per un usuari final, l'arbre final el construiriam sobre tot el set de dades, test inclós. 


Explicacio del classificador

A l'hora de classificar un element donat un arbre el sistema utilitzat és que vaigi recorrent l'arbre escollint la branca a la que respon el seu atribut a la pregunta plantejada de cada node de l'arbre fins a arribar a una fulla. Un cop a fulla consultarà al diccionari de resultats de la fulla el qual donarà una proporció de cada atribut posible i generarem un número aleatori que decidirà (tenenint en compte la proporció) quin resultat final retornarem per aquella fila de dades. 


Decisicio de implementació missing data

Hem decidit que enlloc de construir l'arbre amb la informació faltant i suplir-la en el classificador suplir la informació abans de construir l'arbre. Per fer-ho em separat en valors númerics i valors que no ho són. Per suplir els valors numerics simplement hem agafat la mediana dels valors no faltants de la seva columna i per els valors no numerics hem generat un diccionari de les proporcions dels valors no faltants de la seva columna i generant un valor aleatori tenint en compte aquestes proporcions. 
