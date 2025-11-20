import io;
import sys;
import files;
import string;
import python;

import swift_utils;


string emews_root = getenv("EMEWS_PROJECT_ROOT");
string turbine_output = getenv("TURBINE_OUTPUT");


string find_min =
"""
v <- c(%s)
res <- which(v == min(v))
""";


string count_template =
"""
import get_metrics_norm

drug_name = '%s'
metric = '%s'
instance_dir = '%s'
exp_path = '%s'
params = '%s'

count = get_metrics_norm.curve_comparison(drug_name, metric, instance_dir, exp_path, params)

""";


(string result) get_result(string drug_name, string metric, string instance_dir, string exp_path, string params) {
  printf("Getting result for %s, %s, %s, %s, %s", drug_name, metric, instance_dir, exp_path, params); 
  string code = count_template % (drug_name, metric, instance_dir, exp_path, params);
  result = python_persist(code, "str(count)");
  printf("Result from get_result: %s", result);
}


app (file out, file err) run_model (file shfile, string executable, string param_line, string instance)
{
    "bash" shfile executable param_line emews_root instance @stdout=out @stderr=err;
}

app (void o) summarize_simulation (file summarize_py, string instance_dir, string drug) {
    "python" summarize_py instance_dir drug;
}

app (void o) make_dir(string dirname) {
  "mkdir" "-p" dirname;
}

app (void o) make_output_dir(string instance) {
  "mkdir" "-p" (instance+"/output");
}

// deletes the specified directory
app (void o) rm_dir(string dirname) {
  "rm" "-rf" dirname;
}

app (void o) get_small_summary(string experiments_dir, string output_filename) {
  "bash" "cat" experiments_dir ">>" output_filename;
}

main() {

  printf("Entering main loop");

  string executable = argv("exe");
  string default_xml = argv("settings");
  int num_variations = toint(argv("nv", "3"));
  
  printf("Got arguments: exe=%s, settings=%s, nv=%d", executable, default_xml, num_variations);

  string exp_path = emews_root + "/data/AGS_data/AGS_growth_data/output/csv/";
  string metric = argv("metric");
  string drug_name = argv("drug");
  
  printf("Got paths and params: exp_path=%s, metric=%s, drug_name=%s", exp_path, metric, drug_name);


  file model_sh = input(emews_root + "/scripts/growth_model.sh");
  file upf = input(argv("parameters"));
  file summarize_py = input(emews_root + "/scripts/summarize/summarize_ags_pcdl.py");

  printf("Starting sweep");


  string results[];
  string upf_lines[] = file_lines(upf);
  foreach params,i in upf_lines {
    foreach replication in [0:num_variations-1:1] {
      string instance_dir = "%s/instance_%i_%i/" % (turbine_output, i+1, replication+1);
      // printf("Creating string instance directory");
      // printf("working in %s", instance_dir);

      make_dir(instance_dir) => {        
        file out <instance_dir+"out.txt">;
        file err <instance_dir+"err.txt">;
        string instance_settings = instance_dir + "settings.xml" =>
        // printf("on params2xml...") =>
        params2xml(params, i+replication, default_xml, instance_settings) =>
        // printf("params2xml OK") =>
        (out,err) = run_model(model_sh, executable, instance_settings, instance_dir) => {
          // printf("running model...") =>
          summarize_simulation(summarize_py, instance_dir, drug_name) =>
          results[replication] = get_result(drug_name, metric, instance_dir, exp_path, params) =>
          // results2json(params, instance_dir) =>
          rm_dir(instance_dir + "output/");
          printf("finished sim %s", instance_dir);

        }
      }
    }
  }

  // get_small_summary(turbine_output + "/instance_*/curve_comparison.txt", turbine_output + "/summary.tsv");

}
