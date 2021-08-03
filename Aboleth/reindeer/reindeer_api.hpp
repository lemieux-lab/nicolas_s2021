#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <ctime>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <chrono>
#include <map>
#include <set>
#include <mutex>
#include <ctype.h>
#include <stdio.h>
#include <cstdio>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <omp.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../blight/blight.h"
#include "../blight/zstr.hpp"
#include "../blight/strict_fstream.hpp"
#include "query.hpp"
#include "build_index.hpp"
#include "utils.hpp"
// #include "reindeer.hpp"

#ifndef REN
#define REN

using namespace std;
using namespace chrono;

// Structure used for storing index stuff 
struct Index_Values {
	kmer_Set_Light *ksl;
	uint64_t nb_colors;
	uint k;
	bool record_counts;
	vector<unsigned char*> compr_monotig_color;
	vector<unsigned> compr_monotig_color_sizes;
	string matrix_name;
	long eq_class_nb;
	uint64_t nb_monotig;
	vector<long> position_in_file;
};

extern Index_Values *g_index;
extern uint gFraise;

#ifdef __cplusplus
extern "C" {
#endif
	void *load_index(char *reindeer_dir);
    void *query_on_loaded_index(char *query_path);
    void *all_at_once(char *reindeer_dir);
    uint fraise();
    uint fraise2();
#ifdef __cplusplus
}
#endif

#endif
