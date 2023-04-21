const headlines = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'));
const headlineText = headlines.map((headline) => headline.textContent.trim());
//console.log(headlineText);
const apiUrl = 'http://localhost:5000/predict';
fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ data: headlineText})
    })
    .then((response) => response.json())
    .then((data) => {
      // Get the classification result from the response
      console.log(data)
      let temp={}
      for (let key in data) {
        for(let k2 in data[key]){
          temp[k2]=data[key][k2]
        }
      }
      console.log(temp)
      headlines.forEach(headline => {
        const headlineText = headline.textContent.trim();
        const score = temp[headlineText] || 0; // Default to 0 if no score is found for the headline
        console.log(headlineText, score);
        if (score > 50) {
          headline.style.backgroundColor = 'yellow';
        }
      });
    });