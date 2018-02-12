function init_materialize(){
    $('ul.tabs').tabs();
    $('.collapsible').collapsible();
    $('.chips').material_chip();
    $('.chips-placeholder').material_chip({
        placeholder: 'Add a predictor',
        secondaryPlaceholder: '+predictor',
      });
    $('.chips-autocomplete').material_chip({
        autocompleteOptions: {
        data: {
            'test-autocomplete': null,
        },
        limit: Infinity,
        minLength: 1
        }
    });

    document.getElementById('reset').addEventListener("click", function(){
        post('/api/data/reset').then(function (v) {
            console.log("reset!");
        });
    });
};

