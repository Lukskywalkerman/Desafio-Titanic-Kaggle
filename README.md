🚀 Classificação Sem Código com AWS SageMaker Canvas
Desvendando o Mistério do Titanic 🛳️
<p align="center"> <img src="https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle&logoColor=white" /> <img src="https://img.shields.io/badge/AWS-SageMaker-orange?logo=amazonaws&logoColor=white" /> <img src="https://img.shields.io/badge/Data-CSV-green?logo=file&logoColor=white" /> <img src="https://img.shields.io/badge/Machine%20Learning-Model-success?logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/Artificial%20Intelligence-IA-purple?logo=githubcopilot&logoColor=white" /> </p>

✍️ Artigo DIO: Classificação Sem Código com AWS SageMaker Canvas: Desvendando o Mistério do Titanic

👋 Minha Jornada de ML na DIO

Olá a todos! Meu nome é [Seu Nome], e estou imerso na incrível jornada de Machine Learning através do Bootcamp DIO Nexa - Machine Learning e GenAI na Prática.

Para consolidar o que estamos aprendendo e encorajar a comunidade a colocar a mão na massa, decidi abordar o desafio clássico do Titanic usando uma ferramenta poderosa e acessível: o AWS SageMaker Canvas.

Este artigo é um guia prático, passo a passo, de como construir e interpretar um modelo preditivo. Meu objetivo é mostrar que o ML é acessível a todos.

💡 Mas fica a pergunta: Será que o Jack e a Rose do filme Titanic sobreviveriam ao naufrágio se fossem personagens reais, baseados nas estatísticas? Vamos descobrir juntos!

🚢 Introdução: SageMaker Canvas e o Aprendizado de Máquina (ML)
O naufrágio do Titanic em 1912 é um dos eventos mais notórios da história, e seu registro de passageiros se tornou o dataset mais famoso para quem inicia em Machine Learning (ML).

Neste artigo, vamos usar o AWS SageMaker Canvas para construir um modelo preditivo para este clássico desafio.

O que é o AWS SageMaker Canvas?
O Amazon SageMaker Canvas é uma interface de usuário visual e low-code/no-code dentro do ecossistema AWS.

Ele é projetado para analistas de negócios e usuários que desejam construir modelos de Machine Learning robustos sem precisar escrever código ou ter experiência profunda em ciência de dados.

Principais Vantagens:
📈 Acessibilidade: Democratiza o ML, permitindo que qualquer pessoa utilize o poder dos algoritmos da AWS.

⚡ Velocidade: Automatiza etapas complexas, como a seleção e o ajuste de algoritmos (Auto-ML).

🔍 Interpretabilidade: Fornece relatórios claros de Importância da Variável (Feature Importance) e desempenho do modelo.

1. O Desafio e o Glossário Inicial
O problema do Titanic é um clássico de Classificação Binária, onde a variável alvo (Survived) possui apenas duas classes:

0 = Não Sobreviveu

1 = Sobreviveu

No Canvas, trabalhamos com o conceito de Aprendizado Supervisionado, onde o modelo aprende a partir de um gabarito.

1.1. Glossário Essencial de ML
Termo	Significado
Aprendizado Supervisionado	O modelo aprende a partir de um gabarito (coluna Survived).
Dataset de Treino	Conjunto usado para aprender padrões (train.csv).
Dataset de Teste	Conjunto para avaliar generalização (test.csv).
Outlier	Valor muito distante da maioria (ex: bebê de 6 meses em lista de adultos).
2. Passo a Passo Prático
📂 Dados: Baixe o dataset oficial no Kaggle.

🔗 Importação: Acesse o AWS SageMaker Canvas e carregue o train.csv.

🧹 Limpeza: Para a coluna Age, utilize a imputação pela mediana para preencher valores nulos.

⚠️ Aviso Importante sobre Custos AWS O Amazon SageMaker Canvas oferece um nível gratuito de 2 meses (até 160h/mês). Mais informações: Aprendizado de Máquina sem código - Preços do Amazon SageMaker Canvas - AWS.

3. Treinamento e Meus Resultados Reais
Utilizei o Standard Build e obtive métricas excelentes:

✅ Accuracy: 86,034%

🎯 F1 Score: 82,014%

🔎 Precision: 81,429%

📈 Recall: 82,609%

🏆 AUC-ROC: 0,891

4. Simulação: Jack e Rose sobreviveriam?
📄 Conteúdo do arquivo simulacao_jackRose.csv

csv
PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1000,1,"DeWitt Bukater, Miss. Rose",female,17,1,1,PC 17599,71.2833,B28,C
1001,3,"Dawson, Mr. Jack",male,20,0,0,345063,7.25,,S
💡 Interpretação das colunas:

PassengerId: IDs fictícios.

Pclass: Rose na 1ª classe, Jack na 3ª.

SibSp: 1 para Rose (noivo Cal), 0 para Jack.

Parch: 1 para Rose (mãe Ruth).

Ticket/Fare: coerentes com a classe.

Cabin: Rose em B28, Jack sem cabine.

Embarked: C (Cherbourg) para Rose, S (Southampton) para Jack.

🥁 Veredito da IA:

🌹 Rose → Sobreviveu (1)

🎩 Jack → Não sobreviveu (0)

5. Expandindo Horizontes: Tipos de Problemas em ML
Tipo de Exercício	Objetivo	Exemplo	Métricas
Classificação Multiclasse	Prever entre várias categorias	Gato, Cachorro, Pássaro	Acurácia
Regressão	Prever valor contínuo	Preço de casas	RMSE, MAE
Série Temporal	Prever valores futuros	Vendas mensais	RMSE, MAPE
Clusterização	Agrupar dados sem gabarito	Segmentação de clientes	Silhueta
📣 Chamada para Ação
Agora que você viu o potencial do SageMaker Canvas, te encorajo a replicar o exercício do Titanic e explorar outros tipos de projetos!

👉 Meus próximos passos: aplicar técnicas de Regressão e Série Temporal em dados reais.

📌 Acompanhe meus dados e resultados do Titanic, assim como outros projetos, em meu GitHub: [SEU LINK DO GITHUB AQUI]

Vamos juntos nessa jornada de aprendizado! 🚀
