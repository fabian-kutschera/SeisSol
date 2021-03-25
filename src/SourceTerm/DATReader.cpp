#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <regex>
#include <algorithm>
#include <Eigen/Dense>
#include <math.h>
// Includes to read content of a directory
#include <sys/types.h>
#include <dirent.h>

#include "DATReader.h"



/*
* Given a path, readDAT reads all recevier files and a .ndat file containing the normal vector for each receiver 
* on the surface.
* The scalar pressure field for each receiver is saved. It is obtained by extracting the stress tensor at a given
* position and multiplying it by the normal vector and finally computing the length of the resulting vector.
*
*/
void seissol::sourceterm::readDAT(char const* path, DAT* dat)
{
	// Receiver data files always end on '.dat'
	const std::string ending = ".dat";
	

	// Read content inside path 
	std::vector<std::string> dat_file_list;
	std::string path_str = path;

	DIR* dirp = opendir(path);
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL) {
		std::string current_file_name = dp->d_name;
		if (current_file_name.length() > ending.length() && 
			current_file_name.compare (current_file_name.length() - ending.length(), ending.length(), ending) == 0) 
	    		dat_file_list.push_back(current_file_name);
    
	}
    closedir(dirp);


	for (const auto& filename : dat_file_list) {

		std::string file_path = path_str + "/" + filename;
		
		// rec_nr will hold the number of the current receiver file
		// needed in order to extract the appropriate normal vector
		std::smatch file_name_match;
		std::regex_search(filename, file_name_match, std::regex("-\\d\\d\\d\\d\\d-"));


		// Temporarly save values from current file
		std::vector<double> current_pos;
		std::vector<double> current_time;
		std::vector<Eigen::VectorXd> current_q;

		std::ifstream dat_file(file_path);
		std::string line;

		while (!dat_file.eof())
		{
			getline(dat_file, line, '\n');

			// Lines with position information start with '#'
			if (line.find('#') == 0) {
				std::smatch match;
				// Matches position information of .dat files
				if ( std::regex_search(line, match, std::regex("[-|\\d][0-9.]+e[-+][0-9]+"))) {
					std::string next_pos = match[0];
					current_pos.push_back(std::stod(next_pos));
				}
			// All other important lines start with ' ' (white space)
			} else if ((line.find(' ') == 0)) {

				std::istringstream iss(line);
				std::vector<std::string> tmp;

				for (std::string ex_line; iss >> ex_line; ) {
					tmp.push_back(ex_line);
				}

				current_time.push_back(std::stod(tmp.at(0)));
				
				Eigen::VectorXd q_vector(dat->q_dim);
				
				q_vector(0) = std::stod(tmp.at(1));
				q_vector(1) = std::stod(tmp.at(2));
				q_vector(2) = std::stod(tmp.at(3));
				q_vector(3) = std::stod(tmp.at(4));
				q_vector(4) = std::stod(tmp.at(5));
				q_vector(5) = std::stod(tmp.at(6));

				


				current_q.push_back(q_vector);
			}
		}

		// Save current data 
		dat->pos.push_back(current_pos);
		dat->time.push_back(current_time);
		dat->q.push_back(current_q);

	}

	dat->endtime = dat->time[0].back();
}

/*
 * 	Input:
 *  	- position vector3d
 * 		- timestamp
 * 	Output:
 * 		- Q vector in global Cartesian coordinates
 * 	Interpolation:
 * 		- Linear in time
 * 		- Inverse Weighted Distance (IWD) in space
 */ 

Eigen::VectorXd seissol::sourceterm::DAT::getQ(Eigen::Vector3d const& position, double timestamp)
{
	Eigen::VectorXd zero_vec(q_dim);
	for (int i=0; i<q_dim; ++i)
		zero_vec(i) = 0;
	
	// timestamp < 0 --> past recording time of forward receivers.
	if (timestamp < 0) {
		return zero_vec;
	}

	double pos_x = position(0);
	double pos_y = position(1);
	double pos_z = position(2);
	

	// From all available receivers hold values needed for IWD with n neighbors
	std::vector<int> rec_idx;			
	std::vector<double> rec_distance;

	std::vector<Eigen::VectorXd> q_interpol;
	

	for (int z=0; z < n; ++z) {
		rec_idx.push_back(-1);
		rec_distance.push_back(1e+20);
		q_interpol.push_back(zero_vec);
		
	}

	// Use L2-norm to find the n receivers closest to the current position
	for (int i=0; i < pos.size(); ++i) {
		double l2_dist = ((pos[i][0] - pos_x) * (pos[i][0] - pos_x) + (pos[i][1] - pos_y) * (pos[i][1] - pos_y) 
					+ (pos[i][2] - pos_z) * (pos[i][2] - pos_z));

		int idx_tmp = std::max_element(rec_distance.begin(), rec_distance.end()) - rec_distance.begin();

		if (rec_distance[idx_tmp] > l2_dist) {
			rec_distance[idx_tmp] = l2_dist;
			rec_idx[idx_tmp] = i;
		}
	}


	// Linear interpolation in time
	// Linear interpolation between two known points (x0, y0) and (x1, y1) for a value x:
    // y = y0 + (x - x0) * (y1 - y0) / (x1 - x0) | Here: x = time , y = pressure_field
	for (int i=0; i < n; ++i) {
		auto const lower_it = std::lower_bound(time[rec_idx[i]].begin(), time[rec_idx[i]].end(), timestamp);
		auto const upper_it = std::upper_bound(time[rec_idx[i]].begin(), time[rec_idx[i]].end(), timestamp);

		int lower_idx = lower_it - time[rec_idx[i]].begin();
		int upper_idx = upper_it - time[rec_idx[i]].begin();

		
			if (lower_idx == upper_idx) {
				if (lower_idx == 0) {
					upper_idx = 1;
				} else {
					--lower_idx;
				}
				for (int j=0; j < q_dim; j++) {

					q_interpol[i](j) = q[rec_idx[i]][lower_idx](j) + (timestamp - time[rec_idx[i]][lower_idx]) 
									* (q[rec_idx[i]][upper_idx](j) - q[rec_idx[i]][lower_idx](j))
									/ (time[rec_idx[i]][upper_idx] - time[rec_idx[i]][lower_idx]);
				}
			} else {
				for (int j=0; j < q_dim; j++) {
					q_interpol[i](j) = q[rec_idx[i]][timestamp](j);
				}
			}
	}

	// q_interpol now stores the linearly interpolated values for all n receivers.
	// Apply IWD with n neighbors with power 2
  	
	int idx_tmp = std::min_element(rec_distance.begin(), rec_distance.end()) - rec_distance.begin();

	if (rec_distance[idx_tmp] == 0.0) {
		return q_interpol[idx_tmp];
	} else {

		Eigen::VectorXd numerator(q_dim);
		for (int i=0; i < q_dim; ++i)
			numerator(i) = 0;


		double denominator = 0;
		for (int i=0; i < n; ++i){

			double w_i = 1 / (rec_distance[i] * rec_distance[i]);
			
			for (int z=0; z < q_dim; ++z){
				numerator(z) += q_interpol[i][z] * w_i;
			}

			denominator += w_i;
		}
		Eigen::VectorXd result(q_dim);
		
		for (int w=0; w < q_dim; ++w){
			result(w) = numerator(w) / denominator;
		}
	
		return result;	
	}
}


// int main( int argc, const char* argv[] )
// {

// 	Eigen::MatrixXd test9(9, 9);
	
// 	test9 << 1, 2, 3, 4, 5, 6, 7, 8, 9,
// 		     9, 8, 7, 6, 5, 4, 3, 2, 1, 
// 			 1, 2, 3, 4, 5, 6, 7, 8, 9,
// 		     9, 8, 7, 6, 5, 4, 3, 2, 1,  
// 			 1, 2, 3, 4, 5, 6, 7, 8, 9,
// 		     9, 8, 7, 6, 5, 4, 3, 2, 1,  
// 			 1, 2, 3, 4, 5, 6, 7, 8, 9,
// 		     9, 8, 7, 6, 5, 4, 3, 2, 1,  
// 			 1, 2, 3, 4, 5, 6, 7, 8, 9;

// 	std::cout << "M1: " << std::endl << test9 << std::endl;

// 	Eigen::Map<Eigen::MatrixXd> M2(test9.data(), 6,6);

// 	std::cout << "M2: " << std::endl << M2 << std::endl;

// 	test9.conservativeResize(6, 6);
// 	std::cout << "M1: " << std::endl << test9 << std::endl;



// 	return 0;


// 	seissol::sourceterm::DAT *dat = new seissol::sourceterm::DAT(); 

// 	double time = 2.088; // Expected Result: 1.081081636151335e-01 = 0.1081

// 	// readDAT( "/Users/philippwendland/Documents/TUM_Master/Semester_4/SeisSol_Results/cube_forward/output_7",
// 	// 		 dat );

	
// 	readDAT( "/Users/philippwendland/Documents/TUM_Master/Semester_4/SeisSol_Results/point-source/output-fwd-test/1_05_1-500k",
// 			 dat );

// 	Eigen::Vector3d pos(0, 0.0, -4.99);

// 	Eigen::VectorXd returned_p = dat->getQ(pos, time);

// 	std::cout << "Returned from getQ in main \n";
// 	std::cout << returned_p(0) << "; " << returned_p(1) << "; " << returned_p(2) << "; " << returned_p(3)
// 			  << "; " << returned_p(4) << "; " << returned_p(5) << "\n";


// 	return 0;
// }
