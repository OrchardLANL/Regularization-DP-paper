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

        numlayers = 4
        v_low = 1500.0
        v_high = 3500.0
        pad = 50
        nx = 100 + 2*pad
        nz = 100
        ns = [nx, 1, nz]
        dz = 10
        dx = 10
        mu = 0
        sigma = 1
	function velmodel(j1, j2, x)
                x_uniform = (1/2) .* (1 .+ SpecialFunctions.erf.((x .- mu)/(sigma*sqrt(2))))
                for i = 1:numlayers
                        layer = v_low + 500 * sum(x_uniform[1:i])
                        if j1 <= i*(nz/4) && j1 > (i-1) * (nz/4)
                                return layer
                        end
                end
        end

        d = 3 #dataset sample number
	
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
                        x = randn(numlayers)
                        vel = [velmodel(j1, j2, x) for j1 = 1:nz, j2 = 1:nx]
                        fields[i, :, :] = vel[:, :]
                end
                return nothing
        end
end


@everywhere variablename = "allloghycos"
@everywhere datafilename = "$(results_dir)/trainingdata.jld2"
if !isfile(datafilename)
        numsamples = d
        @time allloghycos = SharedArrays.SharedArray{Float64}(numsamples, ns[3], ns[1]; init=A->samplehyco!(A; setseed=true))
        @time @JLD2.save datafilename allloghycos
end
@JLD2.load datafilename allloghycos

@everywhere vmin = v_low
@everywhere vmax = v_high
pmin = zeros(3)
pmax = zeros(3)

@everywhere Random.seed!(0)
p_trues = Array(SharedArrays.SharedArray{Float64}(3, ns[3], ns[1]; init=samplehyco!))
for i_p in 1:size(p_trues, 1)
                if !isfile("$(results_dir)/opt_$(i_p).jld2")
                        p_true = p_trues[i_p, :, :]
                        nt = 800
                        dt = 0.001
                        function ricker(f, t; Ã=0)  # generate ricker wavelet
                                return (1 .- 2 .* pi.^2 .* f.^2 .* (t.-Ã).^2) .* exp.(-pi.^2 .* f.^2 .* (t.-Ã).^2)
                        end
                        t = range(0, stop=nt*dt, length=nt)
                        wave = 1e5 .*ricker(45, t; Ã=0.06)
                        fr = zeros(nz, nx, nt)  # create forcing term (ricker wavelet)
                        halfx = floor(Int, nx/2)
                        halfz = floor(Int, nz/2)
                        sz = 1              # source z location
                        sx = halfx          # source x location
                        fr[sz, sx, :] = wave
                        fr = reshape(fr, nz * nx, nt)
                        function uwave(p)
                                vel = reshape(p, nz*nx)
                                return  DPFEHM.getuIters(vel, fr, nz, nx, nt, dz, dx, dt)
                        end
                        xloc = Int.(Array(pad:nx-pad))  # padding on either side
                        zloc = Int.(3 .*ones(size(xloc)))
                        ridx = floor.(Int, DPFEHM.linearIndex.(nz.*ones(size(xloc)), zloc, xloc))
                        utrue = uwave(p_true)[ridx, :] # get the shot record
                        function objfunc(p_flat)
                                usave = uwave(p_flat)
                                l2norm = 0
                                for i=1:length(ridx)
                                        l2norm += sum((utrue[i, :] .- usave[ridx[i], :]).^2)
                                end
                                return l2norm / 1e-6
                        end
                        function velocity(x)
                                vel = [velmodel(j1, j2, x) for j1 = 1:nz, j2 = 1:nx]
                                return vel
                        end
                        x0 = zeros(4)
                        options = Optim.Options(iterations=200, extended_trace=false, store_trace=true, show_trace=true, x_tol=1e-8)
                        @time p_flat, opt = RegularizationDP.optimize(objfunc, velocity, x->sum(x .^ 2), x0, options)
                        @JLD2.save "$(results_dir)/opt_$(i_p).jld2" p_true p_flat opt
                        fig, axs = PyPlot.subplots(1, 2)
                        axs[1].imshow(reshape(velocity(opt.minimizer), size(p_true)...), vmin=vmin, vmax=vmax, interpolation="nearest")
                        p = velocity(opt.minimizer)
                        @show minimum(p), maximum(p)
                        ims = axs[2].imshow(p_trues[i_p,:,:], vmin=vmin, vmax=vmax, interpolation="nearest")
                        @show minimum(p_trues[i_p,:,:]), maximum(p_trues[i_p,:,:])
                        wave_inv = uwave(p)[ridx, :]
                        @JLD2.save "$(results_dir)/com_$(i_p).jld2" utrue wave_inv nt
                        fig.colorbar(ims)
                        display(fig)
                        println()
                        fig.savefig("$(results_dir)/result_$(i_p).pdf")
                        PyPlot.close(fig)
                end
		if !isfile("$(results_dir)/wave_$(i_p).jld2")
                	@JLD2.load "$(results_dir)/opt_$(i_p).jld2" p_true p_flat opt
                	p_com = p_true .- p_flat
                	p1 = minimum(p_com)
                	p2 = maximum(p_com)
                	@JLD2.save "$(results_dir)/wave_$(i_p).jld2" p_com p1 p2
        	end
end

alphabet = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"]
fig, axs = PyPlot.subplots(3, 10; figsize=(40, 12))
k = 1
for i_p in 1:size(p_trues, 1)
        global k
                @JLD2.load "$(results_dir)/com_$(i_p).jld2" utrue wave_inv nt
                pt = zeros(nt+2)
                for c = 1:nt+2
                        pt[c] = c*0.001
                end
                pc = zeros(Int, nt)
                q = 0
                for i = 2:10:101
                        q += 1
                        pc[q] = i
                end
                for i in 1:10
                	axs[i_p, i].plot(pt, wave_inv[pc[i], :], "r", alpha=0.7)
			axs[i_p, i].scatter(pt, utrue[pc[i], :], 10, :black)
                        axs[i_p, i].set_title("receiver_$(pc[i])")
                        axs[i_p, i].set_ylabel("Amplitude")
                        axs[i_p, i].set_xlabel("Time (s)\n($(alphabet[k]))")
                        k += 1
                end
end
fig.tight_layout()
display(fig)
fig.savefig("$(results_dir)/W-T.pdf")
println()
PyPlot.close(fig)

for i_p in 1:size(p_trues, 1)
        @JLD2.load "$(results_dir)/wave_$(i_p).jld2" p1 p2
        pmin[i_p] = p1
        pmax[i_p] = p2
end
@show pmin, pmax

alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
fig, axs = PyPlot.subplots(3, 3; figsize = (12, 9))
k = 1
for i_p in 1:size(p_trues, 1)
        global k
                @JLD2.load "$(results_dir)/opt_$(i_p).jld2" p_true
                axs[1, i_p].imshow(p_true, vmin=vmin, vmax=vmax, interpolation="nearest")
                axs[1, i_p].set_aspect("equal")
                axs[1, i_p].set_title("Reference Field")
                axs[1, i_p].set_ylabel(L"y")
                axs[1, i_p].set_xlabel(L"x" * "\n($(alphabet[k]))")
                k += 1
                @JLD2.load "$(results_dir)/opt_$(i_p).jld2" p_flat
                p_opt = reshape(p_flat, size(p_trues[1, : ,:]) ...)
                axs[2, i_p].imshow(p_opt, vmin=vmin, vmax=vmax, interpolation="nearest")
                axs[2, i_p].set_aspect("equal")
                axs[2, i_p].set_title("Relative Error: $(round(sum((p_opt .- p_true) .^ 2) / sum((p_trues[i_p, :, :] .- StatsBase.mean(p_trues[i_p, :, :])) .^ 2); digits=10))")
                axs[2, i_p].set_ylabel(L"y")
                axs[2, i_p].set_xlabel(L"x" * "\n($(alphabet[k]))")
                k += 1
                p_com = p_true[:,:] .- p_opt[:,:]
                axs[3, i_p].imshow(p_com, vmin=-195, vmax=195, interpolation="nearest", cmap="coolwarm") #this is to make "0" in the center of colorbar. The range got from pmin and pmax.
                #axs[3, i_p].imshow(p_com, vmin=minimum(pmin), vmax=maximum(pmax), interpolation="nearest", cmap="coolwarm") 
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
