#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/lapack_full_matrix.h>

#include <iomanip> // @@@
#include <limits> //@@@

#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>

#include <Sacado.hpp>


using namespace dealii;

class MySolver
{
public:
  MySolver(const int poly_degree, const unsigned int refine_global, const unsigned int quad_degree);
  ~MySolver();
  void run();
  
private:
  void solve();
  void compute_F_grad_hess();
  void compute_dk();
  void compute_alpha();
  void compute_alpha_derivs(double alpha, double &dF_dAlpha, double &d2F_dAlpha2);
  void create_480_cells();
  void write_output_file();
  void compute_lagrange_det();
  void contour_plot();
  void init_u0();

  const unsigned int refine_global, quad_degree;

  const double pressure = 500.0, c11 = 1.0e+5, c22 = 1.0e+3, c12 = 1.0e+3,
               eps = 0.1, radius = 1.0;
  double delta = 1.0e-0, delta_max = 1.0e+8;
  unsigned int n_dofs;
  double alpha_k;
  bool verbose = false;
  const unsigned int iter_limit_alpha = 100; // numero maximo de iteracoes para alpha
  const unsigned int iter_limit_sk = 100; // numero maximo de iteracoes para s_k
  const double alpha_tol = 0.0001; // criterio de parada da busca pelo alpha
  
  // com solution_tol = 1e-4 o calculo do determinante apresentava valores diferente
  // dos seus vizinhos e ai o lagrange formava picos, com 1e-7 isso ja nao aconteceu mais.
  // De 1e-7 para 1e-10 nao houve uma grande diferenca
  // os resultados do lagrangeano (o calculo mais sensivel a erros) ficaram bem proximos
  const double solution_tol = 1.0e-10; // criterio de parada da busca pelo s_k

  std::ofstream output_file;

  Triangulation<1> triangulation;
  DoFHandler<1>    dof_handler;

  FE_Q<1> fe;

  //std::vector<Sacado::Fad::DFad<Sacado::Fad::DFad<double>>> solution;
  std::vector<double> solution, prev_solution;
  std::vector<double> grad_F, dk;
  std::vector<std::vector<double>> hess_F;

  // comp_grad, comp_dk, comp_alpha, update + criterio de parada
  std::vector<double> timing{0.,0.,0.,0.};
  // output data
  std::vector<std::vector<double>> output_data;
};


MySolver::MySolver(const int poly_degree, const unsigned int refine_global, const unsigned int quad_degree): 
  refine_global(refine_global),
  quad_degree(quad_degree),
  dof_handler(triangulation),
  fe(poly_degree)
{}

MySolver::~MySolver()
{}

void MySolver::compute_F_grad_hess()
{
  //const bool analytic_diff = false;

  // reset gradiente e hessiana para zero
  grad_F.assign(grad_F.size(), 0.0);
  hess_F.assign(grad_F.size(), grad_F);
  

  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> F_delta, E_h, P_h;
  //std::vector<Sacado::Fad::DFad<Sacado::Fad::DFad<double>>> dofs(n_dofs, 0);

  // inicializacao de F_delta, E_h, ... , dofs
  F_delta = 0.0;
  E_h = 0.0;
  P_h = 0.0;
  /* for (unsigned int i = 0; i < n_dofs; ++i) // diz que dofs sao variaveis independentes
  {
    dofs[i] = solution[i];
    dofs[i].diff(i, n_dofs);
    dofs[i].val().diff(i, n_dofs);
  } */

  const QGauss<1>  quadrature_formula(quad_degree);
  FEValues<1> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);
  
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> sg, sg_prime; // scalar_product(s,g) e (s,g')

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    // se tiver que achar derivadas ja usando os dofs locais precisa desse codigo comentado
    std::vector<Sacado::Fad::DFad<Sacado::Fad::DFad<double>>> dofs(dofs_per_cell);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      dofs[i] = solution[local_dof_indices[i]];
      dofs[i].diff(i, dofs_per_cell);
      dofs[i].val().diff(i, dofs_per_cell);
      //std::cout << dofs[i] << std::endl;
      //std::cout << solution[local_dof_indices[i]] << std::endl;
    }

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      sg = 0.0;
      sg_prime = 0.0;
      double r = fe_values.quadrature_point(q)[0]; // coordenada global do ponto de quadratura
    
      // Calculo (s.g) e (s.g')
      // na verdade teria que passar por todos dofs, mas como phi é sempre zero fora da celula
      // so passamos pelos dofs da celula para calcular o produto escalar
      // lembrar que "g" é uma funcao de r, entao ela vai fazer aprte da somatoria nos quad normalmente
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const double phi_i = fe_values.shape_value(i, q);
        const double phi_i_prime = fe_values.shape_grad(i, q)[0]; // acessa o seu unico elemento
        sg += dofs[i] * phi_i;
        sg_prime += dofs[i] * phi_i_prime;
      }

      E_h = 0;
      //E_h = (pow(rho*sg_prime, 2) + 2*gama*pow(sg,2)) * fe_values.JxW(q); //+=
      E_h = (
        //(sg+1)*(sg+1) // teste: minimizar a funcao (u+1)**2 ok!
        c11*r*pow(sg_prime,2) +
        2.0*c12*sg_prime*sg +
        c22/r*pow(sg,2)
      ) * fe_values.JxW(q);

      //std::cout << "E_h\n";
      //std::cout << E_h << std::endl;
      

      //F_delta = c11/2 * E_h + P_h / delta;
      F_delta = E_h;

      // adiciona contribuicao de uma das parcelas da somatoria
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        grad_F[local_dof_indices[i]] += F_delta.dx(i).val();
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
          hess_F[local_dof_indices[i]][local_dof_indices[j]] += F_delta.dx(i).dx(j);
      }
        
    }
  } // end for cells

  // adiciona a parcela da derivada que nao esta no somatorio de pontos de quadratura (só do eta_n)
  grad_F[n_dofs-1] += 2.0*pressure*radius; // 2.0*c12*solution[n_dofs-1]
  hess_F[n_dofs-1][n_dofs-1] += 0; //2.0*c12;
    
}

void MySolver::compute_dk()
{
  /* std::cout << "DK:\n";
  std::cout << "grad_F:\n";
  for(unsigned int i=0; i < n_dofs; ++i)
    std::cout << grad_F[i] << std::endl;
  std::cout << "hess_F:\n";
  for(unsigned int i=0; i < n_dofs; ++i)
    for(unsigned int j=0; j < n_dofs; ++j)
      std::cout << hess_F[i][j] << std::endl; */
  

  // cria o Tensor que vai conter o gradiente e a hessiana
  //Vector<double> gradT(grad_F.begin(), grad_F.end()), dkT(grad_F.begin(), grad_F.end());
  //dkT.reinit(n_dofs);
  //FullMatrix<double> hessT(n_dofs, n_dofs), inv_hessT(n_dofs, n_dofs);

  // Resolvendo o sistema com solver lapack
  // Na verdade o ideal seria trabalhar com a hessiana como uma SparseMatrix e usar
  // algum solver de matriz esparsa da dealii
  Vector<double> dkT(grad_F.begin(), grad_F.end());
  LAPACKFullMatrix<double> lapack_hess(n_dofs);
  
  for (unsigned int i = 0; i < n_dofs; ++i)
  {
    //gradT(i) = grad_F[i];
    for (unsigned int j = 0; j < n_dofs; ++j)
      lapack_hess(i,j) = hess_F[i][j];
      //hessT[i][j] = hess_F[i][j];
      
  }
  //==========================================
  //SolverControl solver_control(1000, 1e-12);
  //SolverCG<> solver(solver_control); //<> vazio indica que é o padrao: Vector<double>
  //PreconditionSSOR<> preconditioner; //<> vazio indica que é o padrao: SparseMatrix<double>
  //preconditioner.initialize(hessT, 1.2);
  //solver.solve(hessT, dkT, gradT, PreconditionIdentity());
  //==========================================
  
  // .solve espera que ja tenhamos feito a LU factorization
  // inicialmente dkT é o vetor do lado direito, mas apos o .solve ele vira a solucao do sistema
  lapack_hess.compute_lu_factorization();
  lapack_hess.solve(dkT);

  //inv_hessT.invert(hessT);
  //inv_hessT.vmult(dkT, gradT);
  
  // dk[0] nunca sera atualizado, entao dk[0] sera sempre zero, entao new_s[0] e solution[0] serao sempre zero
  for (unsigned int i = 1; i < n_dofs; ++i)
    dk[i] = -dkT(i); // d_k = - inv(hess)*grad

}

void MySolver::compute_alpha_derivs(double alpha, double &dF_dAlpha, double &d2F_dAlpha2)
{
  // alphaAD é uma variavel interna de compute_alpha_derivs que recebe o valor do alpha atual
  // e é usada para fazer as derivadas
  std::vector<Sacado::Fad::DFad<Sacado::Fad::DFad<double>>> new_s(n_dofs, 0.0); // new_s = s_k + alpha^(i) * d_k
  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> alphaAD;
  alphaAD = alpha;
  alphaAD.diff(0,1); // so vamos derivar com relacao a alphaAD
  alphaAD.val().diff(0,1); // mas vamos derivar 2 vezes

  // codigo duplicado (mas modificado) de compute_F_grad_hess:
  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> F_delta, E_h, P_h;
  F_delta = 0.0; E_h = 0.0; P_h = 0.0;
  for (unsigned int i = 0; i < n_dofs; ++i)
    new_s[i] = solution[i] + alphaAD * dk[i];

  const QGauss<1>  quadrature_formula(quad_degree);
  FEValues<1> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);
  
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> sg, sg_prime; // scalar_product(s,g) e (s,g')

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      sg = 0.0;
      sg_prime = 0.0;
      double r = fe_values.quadrature_point(q)[0]; // coordenada global
    
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const double phi_i = fe_values.shape_value(i, q);
        const double phi_i_prime = fe_values.shape_grad(i, q)[0]; // acessa o seu unico elemento
        sg += new_s[local_dof_indices[i]] * phi_i;
        sg_prime += new_s[local_dof_indices[i]] * phi_i_prime;
      }

      //E_h = (pow(rho*sg_prime, 2) + 2*gama*pow(sg,2)) * fe_values.JxW(q); //+=
      E_h += (
        //(sg+1)*(sg+1) // teste: minimizar a funcao (u+1)**2 ok!
        c11*r*pow(sg_prime,2) +
        2.0*c12*sg_prime*sg +
        c22/r*pow(sg,2)
      ) * fe_values.JxW(q);
      
    }
  } // end for cells

  E_h = E_h + 2.0*radius*pressure*new_s[n_dofs-1]; //c12*pow(new_s[n_dofs-1],2)
  F_delta = E_h;
    
  // derivada de F_delta com relacao a alpha
  dF_dAlpha = F_delta.dx(0).val();
  d2F_dAlpha2 = F_delta.dx(0).dx(0);
}

void MySolver::compute_alpha()
{
  double alpha = 0, prev_alpha = 0, // alpha(0) = 0
         dF_dAlpha, d2F_dAlpha2;
  std::vector<double> new_s(n_dofs, 0); // new_s = s_k + alpha^(i) * d_k

  for (unsigned int iter = 0; iter < iter_limit_alpha; ++iter)
  {
    // calcula as derivadas e poe em dF_dAlpha, d2F_dAlpha2
    compute_alpha_derivs(alpha, dF_dAlpha, d2F_dAlpha2);

    prev_alpha = alpha;
    alpha = alpha - dF_dAlpha / d2F_dAlpha2;

    // atualiza new_s com o novo alpha e o mesmo s_k (solution) e d_k
    for (unsigned int i = 0; i < n_dofs; ++i)
      new_s[i] = solution[i] + alpha * dk[i];

    // O codigo do Autoint confere se det_grad(y) > eps, aqui eu nao confiro
    // confere se viola (70) em algum dos nos
    /* ... */

    if(std::abs((alpha - prev_alpha)/alpha) < alpha_tol)
    {
      if(verbose)
        std::cout << "\nsaindo...  alpha update: " << std::abs((alpha - prev_alpha)/alpha) << "\n";
      break; // sai do loop do alpha
    }
      
    if(iter == iter_limit_alpha - 1)
      std::cout << "\n   Aviso: loop do alpha atingiu o limite de iteracoes e foi aceito como alpha final.\n";
  }
  alpha_k = alpha;
}

void MySolver::contour_plot()
{
  //std::vector<double> sol(n_dofs, 0.0);
  double E_h;
  E_h = 0.0;

  // solucao para refine_global = 1 e u_0 = 0
  //sol[0] = 0;
  //sol[1] = -0.00949906;
  //sol[2] = -0.012724;

  // solucao para refine_global = 2 e u_0 = 0
  /* sol[0] = 0;
  sol[1] = -0.00925251;
  sol[2] = -0.0123937;
  sol[3] = -0.0143035;
  sol[4] = -0.0156813; */

  // solucao para refine_global = 1 e u_0 = isotropico
  /* sol[0] = 0.0;
  sol[1] = -0.00951485;
  sol[2] = -0.0119798; */

  const QGauss<1>  quadrature_formula(quad_degree);
  FEValues<1> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);
  
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  double sg, sg_prime; // scalar_product(s,g) e (s,g')

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      sg = 0.0;
      sg_prime = 0.0;
      double r = fe_values.quadrature_point(q)[0]; // coordenada global
    
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const double phi_i = fe_values.shape_value(i, q);
        const double phi_i_prime = fe_values.shape_grad(i, q)[0]; // acessa o seu unico elemento
        sg += solution[local_dof_indices[i]] * phi_i;
        sg_prime += solution[local_dof_indices[i]] * phi_i_prime;
      }

      double v = (c11*r*pow(sg_prime,2) +
        //2.0*c12*sg_prime*sg +
        c22/r*pow(sg,2) )* fe_values.JxW(q);
      std::cout << "E_h += " << v << std::endl;

      std::cout << v << "\n" << sg << "\n" << sg_prime << "\n" << fe_values.JxW(q)
                << "\n" << c11 << "\n" << c22 << r << std::endl;

      E_h += (
        c11*r*pow(sg_prime,2) +
        //2.0*c12*sg_prime*sg +
        c22/r*pow(sg,2)
      ) * fe_values.JxW(q);
    }
  } // end for cells

  E_h = E_h + 2.0*radius*pressure*solution[n_dofs-1] + c12*pow(solution[n_dofs-1],2);
  std::cout << "termo extra " << 2 * radius * pressure * solution[n_dofs-1] + c12*pow(solution[n_dofs-1],2) 
            << std::endl; 
  std::cout << "E_h calculado: " << E_h << std::endl;
}

void MySolver::init_u0()
{
  // solucao caso isotropico
  //std::cout << "init_u0\n";
  for(unsigned int i=0; i<n_dofs; ++i)
  {
    solution[i] = -pressure/(c11+c12)*(1.0/(n_dofs-1)*i);
    //solution[i] = -500.0/(100000.+1000.)*(1./(3.-1.)*i);
    //std::cout << solution[i] << std::endl;
  }
}

void MySolver::solve()
{
  //std::cout << "contour plot: ";
  //contour_plot(); // Checkpoiiiiiint
  //init_u0();
  
  // so para "declarar" t1 e t2
  auto t1 = std::chrono::high_resolution_clock::now();
  auto t2 = std::chrono::high_resolution_clock::now();

  for (unsigned int iter_sk = 0; iter_sk < iter_limit_sk; ++iter_sk)
  {
      t1 = std::chrono::high_resolution_clock::now();
    compute_F_grad_hess(); // usa solution e atualiza grad_F, hess_F
      t2 = std::chrono::high_resolution_clock::now();
      timing[0] += std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

      t1 = std::chrono::high_resolution_clock::now();
    compute_dk(); // usa grad_F e hess_F e atualiza dk
      t2 = std::chrono::high_resolution_clock::now();
      timing[1] += std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

      t1 = std::chrono::high_resolution_clock::now();
    compute_alpha(); // usa solution e dk e atualiza alpha_k
      t2 = std::chrono::high_resolution_clock::now();
      timing[2] += std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    t1 = std::chrono::high_resolution_clock::now();

    // atualiza solution
    prev_solution = solution;
    for (unsigned int i = 0; i < n_dofs; ++i)
      solution[i] = solution[i] + alpha_k * dk[i];

    // criterio de parada para a serie de s_k
    double solution_crit, solution_sum = 0, prev_solution_sum = 0;
    for (unsigned int i = 0; i < n_dofs; ++i)
    {
      solution_sum += std::abs(solution[i]);
      prev_solution_sum += std::abs(prev_solution[i]);
    }
    solution_crit = (solution_sum - prev_solution_sum) / (solution_sum + 1e-10);
    if (solution_crit < solution_tol) break; // sai do loop do s_k

    // ja iterou muitas vezes
    if(iter_sk == iter_limit_sk - 1)
      std::cout << "\n   Aviso: loop do s_k atingiu o limite de iteracoes e foi aceito como s_k final.\n";

    t2 = std::chrono::high_resolution_clock::now();
    timing[3] += std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  }
    
    // insere linha = {delta, dof0, dof1, dof2, ..., dofN}
    output_data.emplace_back(solution);
    //output_data[iter_delta].insert(output_data[iter_delta].begin(), delta);
}

void MySolver::create_480_cells()
{
  std::vector<std::vector<double>> cell_sizes(1);
  std::vector<double> tmp(480, 0.);
  Point<1> origin(0), end(1);
  cell_sizes[0] = tmp;

  for(unsigned int i=0; i<480; ++i)
  {
    if (i<300) cell_sizes[0][i] = 0.07/300;
    else if (i>=300 && i<400) cell_sizes[0][i] = 0.39/100;
    else cell_sizes[0][i] = 0.54/80;
  }
  GridGenerator::subdivided_hyper_rectangle(triangulation, cell_sizes, origin, end, true);

  /* std::cout << "RHOS\n";
    for (const auto &cell : triangulation.active_cell_iterators())
    {
      double rho = cell->vertex(1)(0);
      std::cout << rho << "\n";
    } */
} // end create_480_cells()

void MySolver::write_output_file()
{
  output_file.open("out/sol ref" + std::to_string(refine_global) + ".txt");
  
  // add os rho de cada dof na primeira linha de output_data
  std::vector<double> rhos(n_dofs, 0.);
  int i_rhos = 1; // primeiro rho é zero
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      rhos[i_rhos] = cell->vertex(1)(0);
      i_rhos += 1;
    }
    rhos.insert(rhos.begin(), 0); // para alinhar com os deslocs
    output_data.insert(output_data.begin(), rhos);

  for (unsigned int i = 0; i < size(output_data); ++i)
  {
    for (unsigned int j = 0; j < size(output_data[i]); ++j)
    {
      if(j == size(output_data[i])-1) {
        output_file << output_data[i][j] << std::endl;
      } else {
        output_file << output_data[i][j] << ";";
      }
    }
  }
  output_file.close();
  std::cout << "Arquivo " << "'out/sol ref" << refine_global << ".txt' gerado!" << std::endl;
  
} // end write_output_file()

void MySolver::compute_lagrange_det()
{
  // calculo determinante ao longo do raio
  std::vector<double> dets;
  std::vector<types::global_dof_index> local_dof_indices (2);
  for(const auto &cell : dof_handler.active_cell_iterators())
  {
    cell->get_dof_indices(local_dof_indices);
    //double rho = cell->vertex(1)(0); // rho no meio da celula
    double rho_m = (cell->vertex(1)(0) + cell->vertex(0)(0))*0.5; // rho no meio da celula
    double h = cell->vertex(1)(0) - cell->vertex(0)(0);
    double phi_i_prime = 1 / h;
    double sg_prime = solution[local_dof_indices[0]]*(-phi_i_prime) + solution[local_dof_indices[1]]*phi_i_prime;
    //double sg = solution[local_dof_indices[1]]; //sg de rho
    double sg = solution[local_dof_indices[0]]*0.5 + solution[local_dof_indices[1]]*0.5; //sg de rho_m
    //double det = (1 + sg_prime) * pow(1 + sg/rho, 2);
    double det = (1 + sg_prime) * pow(1 + sg/rho_m, 2);
    dets.emplace_back(det);

    //std::cout << std::setprecision (std::numeric_limits<double>::max_digits10)
    //          << det
    //          << std::endl;
  }
  //std::cout << std::setprecision (std::numeric_limits<double>::max_digits10)
  //            << eps
  //            << std::endl;
  
  std::vector<double> lagranges;
  for(unsigned int i=0; i<size(dets); ++i)
  {
    // para delta = 1e+8
    double lagrange = 1. / (1e+8 * pow(dets[i]-eps, 2));
    lagranges.emplace_back(lagrange);
  }

  dets.insert(dets.begin(), 0); // para alinhar com os rhos
  dets.insert(dets.begin(), 0);
  output_data.emplace_back(dets);

  lagranges.insert(lagranges.begin(), 0); // para alinhar com os rhos
  lagranges.insert(lagranges.begin(), 0);
  output_data.emplace_back(lagranges);

}

void MySolver::run ()
{
  for (unsigned int cycle=0; cycle<1; ++cycle)
  {
    std::cout << "Cycle " << cycle << ':' << std::endl;
    if (cycle == 0)
      {
        if(refine_global == 999) create_480_cells();
        else {
          GridGenerator::hyper_cube(triangulation, 0, radius, /*colorize*/ true);
          triangulation.refine_global(refine_global);
        }
      }
    else
      Assert(false, ExcNotImplemented()); //refine_grid ();

    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells()
              << std::endl;

    dof_handler.distribute_dofs(fe); // garantir numeracao do rho=0 ate o rho=radius
    n_dofs = dof_handler.n_dofs();
    solution.resize(n_dofs, 0); // lembrar que primeiro dof é sempre 0, faço isso deixando sempre dk[0]=0
    //solution = {-0.0001,-0.002,-0.005};
    prev_solution.resize(n_dofs, 0);
    grad_F.resize(n_dofs, 0);
    dk.resize(n_dofs, 0);
    hess_F.resize(n_dofs, grad_F);
    alpha_k = 0;
      
    std::cout << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;
    
    solve();
    //compute_lagrange_det();
    //write_output_file();
    for (unsigned int i = 0; i<n_dofs; ++i)
      std::cout << solution[i] << std::endl;

    std::cout << "timing:" << std::endl << timing[0]  << std::endl << timing[1] 
              << std::endl << timing[2] << std::endl << timing[3] << std::endl;

  std::cout << "contour plot: ";
  contour_plot(); // Checkpoiiiiiint
    
  }
}



int main()
{
  try
    {    
      unsigned int ref;
      std::cout << "ref_global: ";
      std::cin >> ref;
      //int poly_degree, unsigned int refine_global, unsigned int quad_degree): 
      MySolver solver(1,ref,1); // 0 para rodar com "480" elementos // antes usava 3 no 3o argumento
      solver.run();      
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
