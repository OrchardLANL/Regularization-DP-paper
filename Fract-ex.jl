using Distributed
using LaTeXStrings
import Flux
import Optim
using  PyPlot
import StatsBase
import Zygote
@everywhere import GaussianRandomFields

@everywhere begin
	import CUDA
        import DPFEHM
	import Flux
        import GeostatInversion
        import JLD2
        import Random
        import RegAE
	import RegularizationDP
        import SharedArrays
        import Statistics
	import SpecialFunctions
	import DifferentiableBackwardEuler
	using PyPlot

        Random.seed!(myid())
        results_dir = "inverse_results"
        if !isdir(results_dir) && myid() == 1
                mkdir(results_dir)
        end
 
        sidelength = 50.0#m
        thickness = 10.0#m
        mins = [-sidelength, -sidelength, 0]
        maxs = [sidelength, sidelength, thickness]
        ns = [100, 100, 1]
	rns = reverse(ns)
        lefthead = 1.0#m
        righthead = 0.0#m
	coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d(mins, maxs, ns, 1.0)
	volumes = volumes * thickness
	
        function permset(j1, j2, x)
	        thickness = 10.0#m
                matriperm = log(1e-15)#m/s
                wellperm = log(1e-3)#m/s
                alpha = 1.3e-9
                beta = 0.5
                p = 1.8
                rmax = 90
                rmin = 10
                mu = 0
                sigma = 1
                if j1 == 51 && j2 > 11 && j2 < 90
	                return wellperm
                elseif (j1 > 51 || j1 < 51) && (j2 > 24 && j2 < 78)
                        for k = 1:5
	                        u = (1/2)*(1+SpecialFunctions.erf((x[k]-mu)/(sigma*sqrt(2))))
                                tl = ((rmax^(1-p) - rmin^(1-p)) * u + rmin^(1-p))^(1/(1-p))
                                kf = log(alpha*tl^beta/thickness)
                                l = floor(Int64, div(tl, 2))
                                S = 50 - l + 1
                                L = 50 + l
                                ex = tl/2 - l
                                kex = 1/((ex/kf)+((1-ex)/matriperm))
       				if j2 == 25 + (k-1)*13 && j1 >= S && j1 <= L
	                                return kf
                                elseif j2 == 25 + (k-1)*13 && (j1 == S - 1 || j1 == L + 1)
                                        return kex
                                elseif j2 == 25 + (k-1)*13 && (j1 < S - 1 || j1 > L + 1)
                                        return matriperm
                                elseif j2 < 25 + (k-1)*13 && j2 > 25 + (k-2)*13
                                        return matriperm
                                end
			end
		else
                        return matriperm
                end
	end

	d = 3 #dataset sample number
	c = 5 #fracture number
	
	xs = range(mins[1]; stop=maxs[1], length=ns[1])
        ys = range(mins[2]; stop=maxs[2], length=ns[2])
        zs = range(mins[3]; stop=maxs[3])
        function samplehyco!(fields::SharedArrays.SharedArray; setseed=false)
                if nworkers() == 1 || size(fields, 1) == 3#if it is small or there is only one processor break up the chunk more simply
                        if myid() <= 2
                                mychunk = 1:size(fields, 1)
                        else
                                mychunk = 1:0
                        end
                else
                        mychunk = 1 + div((myid() - 2) * size(fields, 1), nworkers()):div((myid() - 1) * size(fields, 1), nworkers())
                end
		
		for i in mychunk
                        if setseed
                                Random.seed!(i)
                        end
			x = zeros(5)
			for i in 1:5
				x[i] = Random.rand(-100:100)/100
			end
			perm = [permset(j1, j2, x) for j1 = 1:ns[2], j2 = 1:ns[1]]
			fields[i, :, :] = perm[:, :]
		end
                return nothing
        end
	boundaryhead(x, y) = (lefthead-righthead) * (x - maxs[1]) / (mins[1] - maxs[1])
        dirichletnodes = Int[]
        dirichletheads = zeros(size(coords, 2))
        for i = 1:size(coords, 2)
                if coords[1, i] == mins[1] || coords[1, i] == maxs[1]
                        push!(dirichletnodes, i)
                        dirichletheads[i] = boundaryhead(coords[1:2, i]...)
                end
        end
end


@everywhere variablename = "allloghycos"
@everywhere datafilename = "$(results_dir)/trainingdata.jld2"
if !isfile(datafilename)
        numsamples = d
        @time allloghycos = SharedArrays.SharedArray{Float32}(numsamples, ns[2], ns[1]; init=A->samplehyco!(A; setseed=true))
        @time @JLD2.save datafilename allloghycos
end
@JLD2.load datafilename allloghycos

@everywhere vmin = log((2e-10)/(4.5e7)) ##4.5e7 is the parameter about gas density, viscosity and it converts hydraulic conducitivity to permeability. 
@everywhere vmax = log((15e-10)/(4.5e7)) ##these vmin, vmax are only for plotting, which exclude the matrix and well. Only foucus on the fractures. 
pmin = zeros(3)
pmax = zeros(3)

@everywhere Random.seed!(0)
p_trues = Array(SharedArrays.SharedArray{Float32}(3, ns[2], ns[1]; init=samplehyco!))
for i_p in 1:size(p_trues, 1)
                if !isfile("$(results_dir)/opt_$(i_p).jld2")
                        p_true = p_trues[i_p, :, :]
			obsindices = [1151, 8851] ## 1151 is the left tip of well and 8851 is the right tip of well. 8851 is also the pumping point 
                        logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))
			sources = zeros(size(coords, 2))
			indsr = 8851
			sources[indsr] = -0.82
			specificstorage = fill(0.001, size(coords, 2))
                        h0 = zeros(size(coords, 2) - length(dirichletnodes))
			for m = 2:ns[1]-1
				h0[(m-2)*100+1:(m-1)*100] .= (lefthead - righthead) * (coords[1,(m-1)*100+1] - maxs[1]) / (mins[1] - maxs[1])
			end
			tt = Vector{Float64}(undef, 1+168*2)
			for i = 2:1+168*2
				tt[i] = 60*60*0.5*(i-1)
			end 
			tt[1] = 0.0
			function gethead(p)
                                loghycos = p
                                if maximum(loghycos) - minimum(loghycos) > 50
                                        return fill(NaN, length(sources))#this is needed to prevent the solver from blowing up if the line search takes us somewhere crazy
                                else
                                        @assert length(loghycos) == length(sources)
                                        neighborhycos = logKs2Ks_neighbors(loghycos)
					p = [neighborhycos; dirichletheads]
					t = tt
					head = DifferentiableBackwardEuler.steps(h0, f_gw, f_gw_u, f_gw_p, f_gw_t, p, t; ftol=1e-4)
					head_1 = head[obsindices[1]-100, :]
					head_2 = head[obsindices[2]-100, :]
					head_3d = cat(head_1, head_2, dims=1)
                                        return head_3d
                                end
                        end
				

                        function unpack(p)
				@assert length(p) == length(neighbors) + size(coords, 2)
				Ks = p[1:length(neighbors)]
				dirichleths = p[length(neighbors) + 1:length(neighbors) + size(coords, 2)]
				return Ks, dirichleths
			end
			function f_gw(u, p, t)
				Ks, dirichleths = unpack(p)
				return -DPFEHM.groundwater_residuals(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, sources, specificstorage, volumes)
			end
			function f_gw_u(u, p, t)
				Ks, dirichleths = unpack(p)
				return -DPFEHM.groundwater_h(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, sources, specificstorage, volumes)
			end
			function f_gw_p(u, p, t)
				Ks, dirichleths = unpack(p)
				J1 = -DPFEHM.groundwater_Ks(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, sources, specificstorage, volumes)
				J2 = -DPFEHM.groundwater_dirichleths(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, sources, specificstorage, volumes)
				return hcat(J1, J2)
			end
			f_gw_t(u, p, t) = zeros(length(u))
			
			head_true = gethead(p_true)
                        function objfunc(p_flat; show_parts=false)
				p = reshape(p_flat, size(p_true)...)
                                head = gethead(p)
                                if show_parts
                                        @show sqrt(sum((head[obsindices] .- head_true[obsindices]) .^ 2) / length(obsindices))
                                        @show sqrt(sum((p_true[p_obsindices] .- p[p_obsindices]) .^ 2) / length(p_obsindices))
                                end
                                return  1e4 * sum((head .- head_true) .^ 2) 
                        end
			function len2logK(x)
				pK =   [permset(j1, j2, x) for j1 = 1:100, j2 = 1:100]
				return pK 
			end 
			x0 = zeros(5)
			options = Optim.Options(iterations=200, extended_trace=false, store_trace=true, show_trace=true, x_tol=1e-8)
                        @time p_flat, opt = RegularizationDP.optimize(objfunc, len2logK, x->sum(x .^ 2), x0, options)
                        @JLD2.save "$(results_dir)/opt_$(i_p).jld2" p_true p_flat opt
			fig, axs = PyPlot.subplots(1, 3)
                        axs[1].imshow(reshape(len2logK(opt.minimizer), size(p_true)...), vmin=vmin, vmax=vmax, cmap="jet", extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest")
                        p = len2logK(opt.minimizer)
                        @show minimum(p), maximum(p)
                        axs[2].imshow(p_trues[i_p,:,:], vmin=vmin, vmax=vmax, cmap="jet", extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest")
			@show minimum(p_trues[i_p,:,:]), maximum(p_trues[i_p,:,:])
			p_com = reshape(p_trues[i_p,:,:], size(p_true)...)[:,:] .- reshape(len2logK(opt.minimizer), size(p_true)...)[:, :]
			@show minimum(p_com), maximum(p_com)
			p1 = minimum(p_com)
			p2 = maximum(p_com)
                        @JLD2.save "$(results_dir)/com_$(i_p).jld2" p1 p2
			ims = axs[3].imshow(p_com, cmap="jet", extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest")
			fig.colorbar(ims)
                        display(fig)
                        println()
                        fig.savefig("$(results_dir)/result_$(i_p).pdf")
                        PyPlot.close(fig)
                end

		if !isfile("$(results_dir)/head_$(i_p).jld2")
                	@JLD2.load "$(results_dir)/opt_$(i_p).jld2" p_true p_flat opt
                	head_true = gethead(p_true)
                	head_true1 = head_true[1:length(tt)]
                	head_true2 = head_true[length(tt)+1:2*length(tt)]
                	head_inv = gethead(p_flat)
                	head_inv1 = head_inv[1:length(tt)]
                	head_inv2 = head_inv[length(tt)+1:2*length(tt)]
                	ttp = tt[:]/3600
                	@JLD2.save "$(results_dir)/head_$(i_p).jld2" head_true1 head_true2 head_inv1 head_inv2 ttp
        	end
end

for i_p in 1:size(p_trues, 1)
	@JLD2.load "$(results_dir)/com_$(i_p).jld2" p1 p2
	pmin[i_p] = p1
	pmax[i_p] = p2
end 
@show pmin, pmax

alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
fig, axs = PyPlot.subplots(3, 3; figsize = (12, 12))
k = 1
for i_p in 1:size(p_trues, 1)
        global k
                @JLD2.load "$(results_dir)/opt_$(i_p).jld2" p_true
		p_true = p_true .- log(4.5e7)
                axs[1, i_p].imshow(p_true, vmin=vmin, vmax=vmax, extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest", cmap="jet")
                axs[1, i_p].set_aspect("equal")
                axs[1, i_p].set_title("Reference Field")
                axs[1, i_p].set_ylabel(L"y")
                axs[1, i_p].set_xlabel(L"x" * "\n($(alphabet[k]))")
                k += 1
                @JLD2.load "$(results_dir)/opt_$(i_p).jld2" p_flat
                p_opt = reshape(p_flat, size(p_trues[1, : ,:]) ...)
		p_opt = p_opt .- log(4.5e7)
                axs[2, i_p].imshow(p_opt, vmin=vmin, vmax=vmax, extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest", cmap="jet")
                axs[2, i_p].set_aspect("equal")
                axs[2, i_p].set_title("Relative Error: $(round(sum((p_opt .- p_true) .^ 2) / sum((p_trues[i_p, :, :] .- StatsBase.mean(p_trues[i_p, :, :])) .^ 2); digits=10))")
                axs[2, i_p].set_ylabel(L"y")
                axs[2, i_p].set_xlabel(L"x" * "\n($(alphabet[k]))")
                k += 1
		p_com = p_true[:,:] .- p_opt[:,:]
		axs[3, i_p].imshow(p_com, vmin=minimum(pmin), vmax=maximum(pmax), extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest", cmap="seismic")
		axs[3, i_p].set_aspect("equal")
		axs[3, i_p].set_title("Comparison")
		axs[3, i_p].set_ylabel(L"y")
		axs[3, i_p].set_xlabel(L"x" * "\n($(alphabet[k]))")
		k += 1

end
fig.tight_layout()
display(fig)
fig.savefig("$(results_dir)/megaplot2.pdf")
println()
PyPlot.close(fig)

fig, axs = PyPlot.subplots(1, 3; figsize=(12, 4))
k = 1
for i_p in 1:size(p_trues, 1)
        global k
                @JLD2.load "$(results_dir)/opt_$(i_p).jld2" opt
                axs[i_p].semilogy(map(t->t.iteration, opt.trace), map(t->t.value, opt.trace), lw=3, alpha=0.5)
        axs[i_p].set_ylabel("Objective Function")
        axs[i_p].set_xlabel("Iteration\n($(alphabet[k]))")
        k += 1
end
fig.tight_layout()
display(fig)
fig.savefig("$(results_dir)/convergence.pdf")
println()
PyPlot.close(fig)

alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
fig, axs = PyPlot.subplots(1, 3; figsize=(12, 4))
k = 1
for i_p in 1:size(p_trues, 1)
        global k
                @JLD2.load "$(results_dir)/head_$(i_p).jld2" head_true1 head_true2 head_inv1 head_inv2 ttp
                        true1 = head_true1[1:10:end]
                        true2 = head_true2[1:10:end]
                        ti = ttp[1:10:end]
                axs[i_p].scatter(ti, true1*985.6/1e6, 8, :black)
                axs[i_p].scatter(ti, true2*985.6/1e6, 8, :red)
                axs[i_p].plot(ttp, head_inv1*985.6/1e6, "k", alpha=0.5)
                axs[i_p].plot(ttp, head_inv2*985.6/1e6, "r", alpha=0.5)
                axs[i_p].set_ylabel("Pressure (MPa)")
                axs[i_p].set_xlabel("Time (hr)\n($(alphabet[k]))")
                k += 1
end
fig.tight_layout()
display(fig)
fig.savefig("$(results_dir)/P-T.pdf")
println()
PyPlot.close(fig)

