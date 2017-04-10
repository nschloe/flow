# -*- coding: utf-8 -*-
#
'''
Stabilization techniques for PDEs with dominating convection.
For an overview of methods, see

   On spurious oscillations at layers diminishing (SOLD) methods
   for convection-diffusion equations: Part I - A review;
   V. John, P. Knobloch;
   Computer Methods in Applied Mechanics and Engineering,
   Volume 196, Issues 17-20, 15 March 2007, Pages 2197-2215;
   <https://www.sciencedirect.com/science/article/pii/S0045782506003926>.

Also:

   Stability of the SUPG finite element method for transient
   advection-diffusion problems;
   Bochev, Gunzburger, Shadid;
   Comput. Methods Appl. Mech. Engrg. 193 (2004) 2301-2323;
   <http://people.sc.fsu.edu/~mgunzburger/files_papers/gunzburger-stab3.pdf>,
   <http://www.cs.sandia.gov/~pbboche/papers_pdf/2004CMAME.pdf>.

'''
from dolfin import Expression


def supg2(convection, diffusion_coefficient):
    cppcode = '''#include <dolfin/mesh/Vertex.h>

class SupgStab : public Expression {
public:
double sigma;
int p;
boost::shared_ptr<GenericFunction> convection;
boost::shared_ptr<Mesh> mesh;

SupgStab(): Expression()
{}

void eval(Array<double>& tau,
          const Array<double>& x,
          const ufc::cell& c
          ) const
{
  Array<double> v(x.size());
  convection->eval(v, x, c);
  double conv_norm = 0.0;
  for (uint i = 0; i < v.size(); ++i)
    conv_norm += v[i]*v[i];
  conv_norm = sqrt(conv_norm);

  if (conv_norm > DOLFIN_EPS)
  {
    Cell cell(*mesh, c.index);

    // Compute the directed diameter of the cell, cf.
    //
    //    On spurious oscillations at layers diminishing (SOLD) methods
    //    for convection-diffusion equations: Part II - Analysis for P1 and Q1
    //    finite elements;
    //    Volker John, Petr Knobloch;
    //    Comput. Methods Appl. Mech. Engrg. 197 (2008) 1997-2014.
    //
    // The diameter in a direction s is defined as
    //
    //    diam(cell, s) = 2*||s|| / sum_{nodes n_i} |s.\grad\psi|
    //
    // where \psi is the P_1 basis function of n_i.
    //
    const double area = cell.volume();
    const unsigned int* vertices = cell.entities(0);
    assert(vertices);

    double sum = 0.0;
    for (int i=0; i<3; i++)
    {
      for (int j=i+1; j<3; j++)
      {
        // Get edge coords.
        const dolfin::Vertex v0(*mesh, vertices[i]);
        const dolfin::Vertex v1(*mesh, vertices[j]);
        const Point p0 = v0.point();
        const Point p1 = v1.point();
        const double e0 = p0[0] - p1[0];
        const double e1 = p0[1] - p1[1];

        // Note that
        //
        //     \grad\psi = ortho_edge / edgelength / height
        //               = ortho_edge / (2*area)
        //
        // so
        //
        //   (v.\grad\psi) = (v.ortho_edge) / (2*area).
        //
        // Move the constant factors out of the summation.
        //
        // It would be really nice if we could just do
        //    edge.dot((-v[1], v[0]))
        // but unfortunately, edges just dot with other edges.
        sum += fabs(e1*v[0] - e0*v[1]);
      }
    }
    const double h = 4 * conv_norm * area / sum;

    //// The alternative for the lazy:
    //const double h = cell.diameter();

    // Just a little sanity check here.
    const double eps = 1.0e-12;
    if (h > cell.diameter() + eps)
    {
        std::cout << "The directed diameter h (" << h << ") "
                  << "should not be larger than the actuall cell diameter "
                  << "(" << cell.diameter() << ")."
                  << std::endl;
    }
    assert(h < cell.diameter() + eps);

    const double Pe = 0.5*conv_norm * h/(p*sigma);
    assert(Pe > 0.0);

    // Evaluate 1/tanh(Pe) - 1/Pe.
    // If Pe is small, then we're running into serious round-off trouble
    // here: subtracting two large floats between which the difference is
    // small. In this case, use the Taylor expansion around 0.
    const double xi =
      (Pe > 1.0e-6) ?
      1.0/tanh(Pe) - 1.0/Pe :
      Pe/3.0 - pow(Pe, 3)/45.0 + 2.0/945.0 * pow(Pe, 5);

    tau[0] = 0.5*h*xi / (p*conv_norm);

    //if (tau[0] > 1.0e3)
    //{
    //  std::cout << "tau   = " << tau[0] << std::endl;
    //  std::cout << "||b|| = " << conv_norm << std::endl;
    //  std::cout << "Pe    = " << Pe << std::endl;
    //  std::cout << "h     = " << h << std::endl;
    //  std::cout << "xi    = " << xi << std::endl;
    //  //throw 1;
    //}
  }
  else
  {
    tau[0] = 0.0;
  }

  return;
}
};
'''
    # TODO set degree
    tau = Expression(cppcode)
    tau.convection = convection
    tau.mesh = convection.function_space().mesh()
    tau.sigma = diffusion_coefficient
    tau.p = convection.function_space().element_degree()

    return tau
