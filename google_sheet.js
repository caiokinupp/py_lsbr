// Primeira coluna da planilha são totos os tickers copiados do site da B3
// Vá em extensões -> App Script e copie o código abaixo
// O resultado vai ser a lista na syntaxe do python

function transformarColunaEmLista() {
    // Abre a planilha ativa
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    
    // Define o intervalo da coluna que você quer transformar em lista
    // Aqui estamos considerando a coluna A, da linha 1 até a última linha com dados
    var range = sheet.getRange("A1:A" + sheet.getLastRow());
    
    // Obtém os valores do intervalo como uma matriz 2D
    var values = range.getValues();
    
    // Transforma a matriz 2D em uma lista de strings, adicionando aspas
    var listaDeStrings = values.map(function(row) {
      return '"' + String(row[0]) + '"';  // Adiciona aspas ao redor de cada valor
    });
  
    // Exibe a lista no console do Apps Script
    Logger.log(listaDeStrings);
    
    // Opcional: Retorna a lista, caso queira usá-la em outra função
    return listaDeStrings;
  }
  