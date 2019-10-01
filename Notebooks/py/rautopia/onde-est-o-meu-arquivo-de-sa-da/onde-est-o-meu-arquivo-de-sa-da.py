#!/usr/bin/env python
# coding: utf-8

# Obs:Este artigo é uma tradução de.
# <br>
# <br>
# === Por favor, leia também os comentários abaixo, no caso Kagglers escrever atualizações para esta posiçãot ===*
# <br>
# <br>
# 
# Se você é novo no Kaggle, você pode se perguntar como criar um arquivo de saída. Talvez você já tenha executado em seu notebook (interface parecida com jupyter notebook) uma função como .to_csv, mas   **você não vê seu arquivo em qualquer lugar **?  Eu tive o mesmo problema.  Você precisa submeter seu notebook **. Há um botão Confirmar no canto superior direito do painel principal do seu notebook.
# 
# <br>
# <br>Para criar um arquivo a partir do zero, passo a passo, leia ou forkeie e execute este notebook.
# <br>
# <br>Digamos que você tenha iniciado seu primeiro kernel baseado no banco de dados Titanic, indo ao <a href="https://www.kaggle.com/c/titanic/kernels">Kernels</a> separador e clicando no botão Novo Kernel. Você veria algo assim:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # algebra linear
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Quaisquer resultados que você gravar no diretório atual serão salvos como saída.


# Você vai ler o arquivo de entrada do conjunto de teste, fazer uma previsão muito bruta (uma regra simples "todas as mulheres sobreviveram, nenhum homem sobreviveram") e criar um dataframe simples com resultados que você gostaria de enviar.

# In[ ]:


test = pd.read_csv('../input/test.csv')
test['Survived'] = 0
test.loc[test['Sex'] == 'female','Survived'] = 1
data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':test['Survived']
})


# Agora que você tem seu dataframe, você gostaria de exportá-lo como um arquivo csv, assim:

# In[ ]:


data_to_submit.to_csv('csv_to_submit.csv', index = False)


# Tudo funciona perfeitamente, mas o problema é que você não pode ver o seu arquivo em qualquer lugar nesta página, nem no seu perfil, guia Kernels, em nenhum lugar! 
# Isso é porque você não submeteu seu notebook ainda. Faça isso, **click o botão "Commit" ** - enquanto escrevo, este é um botão azul claro no canto superior direito da página do meu caderno, no painel principal. (
# Há também um painel direito com Sessions, Versions etc. Você pode ignorá-lo por enquanto). Pode levar um minuto para o servidor Kaggle publicar seu notebook.
# <br>
# <br>Quando esta operação estiver concluída, você pode voltar clicando '<<' botão no canto superior esquerdo.Então você deve ver o seu notebook com uma barra superior que tem algumas abas: Notebook, Code, Data, **Output**, Comments, Log ... Edit Notebook.
# Clique na aba "output". Você deve ver o arquivo CSV de saída lá, pronto para download!
