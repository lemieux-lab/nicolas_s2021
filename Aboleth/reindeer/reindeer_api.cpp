#include "reindeer_api.hpp"


using namespace std;
using namespace chrono;

Index_Values *g_index = NULL;
uint gFraise = 0;

extern "C" void *load_index(char *reindeer_dir) {
    std::string dir_str = std::string(reindeer_dir);
	string color_load_file;
	string color_dump_file("");
	string matrix_name;
	string fof;
	long eq_class_nb;
	bool record_counts(0);
	
	uint64_t nb_colors, nb_monotig ;
	uint k, record_option;
	vector<unsigned char*> compr_monotig_color;
	vector<unsigned> compr_monotig_color_sizes;

	color_load_file = getRealPath("reindeer_matrix_eqc.gz", dir_str);
	size_t wo_ext = color_load_file.find_last_of("."); 
	matrix_name = color_load_file.substr(0, wo_ext);
	
	read_info(k, nb_monotig, eq_class_nb, nb_colors, record_option, matrix_name);
	if (record_option == 1)
	{
		record_counts = true;
	}
	
	std::ofstream index_loading_semaphore(dir_str + "/index_loading"); // begin semaphore
	//~ long eq_class_nb(0);
	bool quantize = false, log = false;
	kmer_Set_Light* ksl = load_rle_index(k, color_load_file, color_dump_file, fof, record_counts,  1,  dir_str, compr_monotig_color, compr_monotig_color_sizes, false, eq_class_nb, nb_colors, quantize, log);
	vector<long> position_in_file;
	string position_file_name(matrix_name + "_position.gz");
	get_position_vector_query_disk(position_in_file, position_file_name, nb_monotig);
	// index_values to_return =  index_values(ksl, nb_colors, k, record_counts, compr_monotig_color, compr_monotig_color_sizes, matrix_name, eq_class_nb, nb_monotig, position_in_file);
	// return to_return;
	g_index = new Index_Values();
	g_index->ksl = ksl;
	g_index->nb_colors = nb_colors;
	g_index->k = k;
	g_index->record_counts = record_counts;
	g_index->compr_monotig_color = compr_monotig_color;
	g_index->compr_monotig_color_sizes = compr_monotig_color_sizes;
	g_index->matrix_name = matrix_name;
	g_index->eq_class_nb = eq_class_nb;
	g_index->nb_monotig = nb_monotig;
	g_index->position_in_file = position_in_file;
// matrix_name, eq_class_nb, nb_monotig, position_in_file};
    // g_index = &index;
    // cout << g_index << endl;
    // cout << g_index->k << endl;
    // return g_index;
}

extern "C" void *query_on_loaded_index(char *kmer, int *to_build) {
    // cout << g_index << endl;
    // cout << g_index->k << endl;
	if(g_index == NULL) {
		cerr << "No index has been loaded! Use load_reindeer_index() first!" << endl;
		return 0;
	}
	vector<vector<uint32_t>> query_unitigID(g_index->nb_colors,{0});
	std::string input = std::string(kmer);
	// cout << input.empty() << endl;
	// cout << exists_test(input) << endl;
	// std::string name = std::string("test");
	api_query(input, to_build, *(g_index->ksl), g_index->nb_colors, g_index->k, g_index->record_counts, 40, query_unitigID, 1, g_index->compr_monotig_color, g_index->compr_monotig_color_sizes, false, g_index->matrix_name, g_index->eq_class_nb, g_index->nb_monotig, g_index->position_in_file);
	// doQuery(input, name, *(g_index->ksl), g_index->nb_colors, g_index->k, g_index->record_counts, 40, query_unitigID, 4, g_index->compr_monotig_color, g_index->compr_monotig_color_sizes, false, g_index->matrix_name, g_index->eq_class_nb, g_index->nb_monotig, g_index->position_in_file);
	// cout << to_return << endl;
}

extern "C" void *all_at_once(char *reindeer_dir) {
    load_index(reindeer_dir);
    // query_on_loaded_index("test");

}

extern "C" uint fraise2() {
    gFraise++;
    cout << gFraise << endl;
    cout << &gFraise << endl;
    return gFraise;
}

extern "C" uint fraise() {
    gFraise++;
    cout << gFraise << endl;
    cout << &gFraise << endl;
    return gFraise;
}

extern "C" void test_table(int to_add, int *table) {
	// cout << table.size() << endl;
	cout << table << endl;
	table[0] = 42;
	cout << table[2] << endl;
	// cout << table[1] << endl;
}