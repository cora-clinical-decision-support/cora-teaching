function display_test_select(select_test) {
    // expects a bool
    if (document.getElementById('mod-test').hasAttribute("select_test") !== select_test) {
        console.log("Needs to update status");

        if (select_test) {
            console.log("active");
            document.getElementById('mod-test').setAttribute("select_test", "active");

        } else {
            console.log("inactive");
            document.getElementById('mod-test').removeAttribute("select_test");
        }
    }
}


function flip_test_select(_this_row_index) {
    console.log("tring to flip a test case!");

    // // check if this patient has been selected
    // status = document.getElementById('mod-test').hasAttribute('test-select'):

    // // flip test case status
    // if (status){
    //     document.getElementById('mod-test').removeAttribute(
    // }


    post('/api/tests/set', { 'index': _this_row_index }).then(function (v) {
        // console.log(v.result);
        display_test_select(v.result['select_test']);
    });
}

function update_case(_this_row_index) {
    // display test-select status
    display_test_select(_this_row_index);

    // display data on the case sheet
    post('/api/data/row', { "row_index": _this_row_index }).then(function (f) {

        // returns all features of this patient (json obj)
        _this_row = JSON.parse(f.result)[0];

        document.getElementById('BLOOD_TYPE').innerHTML = _this_row.BLOOD_TYPE;

        // abbreviate DEVICE_STRATEGY
        document.getElementById('DEVICE_STRATEGY').innerHTML = _this_row.DEVICE_STRATEGY;
        document.getElementById('AGE_GENDER').innerHTML = _this_row.AGE_GRP + _this_row.GENDER;
        document.getElementById('NYHA').innerHTML = 'NYHA ' + _this_row.NYHA;
        document.getElementById('ALBUMIN_G_L').innerHTML = _this_row.ALBUMIN_G_L;

        // labs
        document.getElementById('INR').innerHTML = _this_row.INR;
        document.getElementById('HGB').innerHTML = _this_row.MIN_HGB + '~' + _this_row.MAX_HGB;
        document.getElementById('CREAT_UMOL_L').innerHTML = _this_row.CREAT_UMOL_L;
        document.getElementById('ALBUMIN_G_L').innerHTML = _this_row.ALBUMIN_G_L; //ALB
        document.getElementById('WBC').innerHTML = _this_row.WBC_X10_9_L;
        document.getElementById('SGPT').innerHTML = _this_row.SGPT_ALT;
        document.getElementById('SGOT').innerHTML = _this_row.SGOT_AST;

        // echo
        document.getElementById('LVEF').innerHTML = _this_row.LVEF;

        // others
        document.getElementById('walk').innerHTML = _this_row.SIX_MIN_WALK;

        PRIMARY_DGN_dict = {
            1: 'Cancer',
            2: 'Congenital Heart Disease',
            3: 'Coronary Artery Disease',
            4: 'Dilated Myopathy: Adriamycin',
            5: 'Dilated Myopathy: Alcoholic',
            6: 'Dilated Myopathy: Familial',
            7: 'Dilated Myopathy: Idiopathic',
            8: 'Dilated Myopathy: Ischemic',
            9: 'Dilated Myopathy: Myocarditis',
            10: 'Dilated Myopathy: Other Specify',
            11: 'Dilated Myopathy: Post partum',
            12: 'Dilated Myopathy: Viral',
            13: 'Hypertrophic Cardiomyopathy',
            14: 'Restrictive Myopathy: Amyloidosis',
            15: 'Restrictive Myopathy: Endocardial Fibrosis',
            16: 'Restrictive Myopathy: Idiopathic',
            17: 'Restrictive Myopathy: Other specify',
            18: 'Restrictive Myopathy: Sarcoidosis',
            19: 'Restrictive Myopathy: Sec to Radiation/Chemotherapy',
            20: 'Valvular Heart Disease',
            21: 'None',
            51: 'Congenital Heart Disease: Biventricular: CAVC/VSD/ASD',
            52: 'Congenital Heart Disease: Biventricular: Congenitally Corrected Transposition (I-TGA)(CC-TGA)',
            53: 'Congenital Heart Disease: Biventricular: Ebsteins Anomaly',
            54: 'Congenital Heart Disease: Biventricular: Kawasaki Disease',
            55: 'Congenital Heart Disease: Biventricular: Left Heart Valve/Structural Hypoplasia',
            56: 'Congenital Heart Disease: Biventricular: TOF/TOF Variant#',
            57: 'Congenital Heart Disease: Biventricular: Transposition of the Great Arteries (d-TGA)',
            58: 'Congenital Heart Disease: Biventricular: Truncus Arteriosus',
            59: 'Congenital Heart Disease: Single Ventricle: Heterotaxy/Complex CAVC',
            60: 'Congenital Heart Disease: Single Ventricle: Hypoplastic Left Heart',
            61: 'Congenital Heart Disease: Single Ventricle: Other',
            62: 'Congenital Heart Disease: Single Ventricle: Pulmonary Atresia with IVS',
            63: 'Congenital Heart Disease: Single Ventricle: Pulmonary Atresia with IVS (RVDC)',
            64: 'Congenital Heart Disease: Single Ventricle: Unspecified',
            998: 'Unknown'
        }
        document.getElementById('PRIMARY_DGN').innerHTML = PRIMARY_DGN_dict[_this_row.PRIMARY_DGN];

        // calculate BMI
        // reference: https://stackoverflow.com/questions/21698044/basic-bmi-calculator-html-javascript
        this_BMI = (_this_row.WGT_KG) / Math.pow(_this_row.HGT_CM / 100, 2);
        document.getElementById('BMI').innerHTML = 'BMI' + (Math.round(this_BMI * 1) / 1).toString();


    });

};