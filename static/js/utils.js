var baseUrl = "http://localhost:8080";

// call this function for GET services
function get(url) {
  var promise = new Promise((resolve, reject) => {
    fetch(baseUrl + url, {
      method: "get",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        "Data-Type": "json"
      }
    })
      .then(function(response) {
        resolve(response.json());
      })
      .catch(err => {
        reject(err);
      });
  });

  return promise;
}

// call this function for POST services
function post(url, object) {
  var promise = new Promise((resolve, reject) => {
    fetch(baseUrl + url, {
      method: "post",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        "Data-Type": "json"
      },
      body: JSON.stringify(object)
    })
      .then(function(response) {
        resolve(response.json());
      })
      .catch(err => {
        reject(err);
      });
  });

  return promise;
}
