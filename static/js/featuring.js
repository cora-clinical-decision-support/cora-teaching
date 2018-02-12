function pop_feature_list(activeColNames){
    // para: a list of column names (features)
    // populates a list of feature checkboxes (unchecked)
    for (var i = 0; i < activeColNames.length; i++) {
        var fea_iter = document.createElement("p");
        fea_iter.innerHTML = `<input type='checkbox' id='feature-${i}'/><label for='feature-${i}'>${activeColNames[i]}</label>`;
        document.getElementById('feature-table').appendChild(fea_iter);
        //"<p><input type='checkbox' id='feature-${i}'/><label for='feature-${i}'>${activeColNames[i]}</label></p>"
    };
};

function update_features(activeColNames){
    // para: a list of column names (active features)
    // output: add chips and tick checkboxes of active features

    $('#feature-chips').material_chip({
        data: activeColNames.map(v => ({tag: v}))
    });

    for (var i = 0; i < activeColNames.length; i++) {
        // console.log('#feature-' + (i).toString(), activeColNames[i]);
        // check checkboxes for active features
        $('#feature-' + (i).toString()).attr("checked", "checked");
    }
};