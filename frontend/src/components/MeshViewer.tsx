/**
 * Vanilla three.js mesh viewer. No drei / no react-three-fiber so it works
 * cleanly under React 19. Loads a .ply, centers and normalizes it, gives
 * the user OrbitControls + a faint grid floor + a directional light.
 */
import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

import { cn } from "@/lib/utils";

export function MeshViewer({ url, className }: { url: string; className?: string }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !url) return;

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    scene.fog = new THREE.Fog(0x0a0a0a, 18, 35);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.05, 200);
    camera.position.set(5, 4, 5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.35));
    const sun = new THREE.DirectionalLight(0xffffff, 0.95);
    sun.position.set(8, 12, 6);
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0x88aaff, 0.25);
    fill.position.set(-6, 4, -5);
    scene.add(fill);

    const grid = new THREE.GridHelper(40, 40, 0x2a2a2a, 0x161616);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.7;
    scene.add(grid);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0.5, 0);

    let mesh: THREE.Mesh | null = null;
    const loader = new PLYLoader();
    loader.load(
      url,
      (geom) => {
        geom.computeVertexNormals();
        geom.center();
        // Auto-scale: fit longest dim to ~4 world units
        const bbox = new THREE.Box3().setFromBufferAttribute(
          geom.getAttribute("position") as THREE.BufferAttribute,
        );
        const size = bbox.getSize(new THREE.Vector3());
        const longest = Math.max(size.x, size.y, size.z);
        const scale = longest > 0 ? 4 / longest : 1;
        const material = new THREE.MeshStandardMaterial({
          color: 0xdcdcdc,
          roughness: 0.55,
          metalness: 0.05,
          side: THREE.DoubleSide,
        });
        mesh = new THREE.Mesh(geom, material);
        mesh.scale.setScalar(scale);
        mesh.position.y = (size.y * scale) / 2;
        scene.add(mesh);
      },
      undefined,
      (err) => {
        console.error("PLY load failed", err);
      },
    );

    let rafId = 0;
    const tick = () => {
      rafId = requestAnimationFrame(tick);
      controls.update();
      renderer.render(scene, camera);
    };
    tick();

    const ro = new ResizeObserver(() => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (!w || !h) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    ro.observe(container);

    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      if (mesh) {
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
      }
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [url]);

  return (
    <div
      className={cn(
        "relative w-full h-full border border-border rounded-sm overflow-hidden bg-[oklch(0.06_0_0)]",
        className,
      )}
    >
      <div ref={containerRef} className="absolute inset-0" />
      <div className="absolute pointer-events-none top-3 right-3 mono text-[10px] text-muted-foreground/70 px-2 py-1 rounded-sm bg-background/60 backdrop-blur-sm">
        drag <span className="text-foreground">orbit</span> · shift+drag{" "}
        <span className="text-foreground">pan</span> · wheel{" "}
        <span className="text-foreground">zoom</span>
      </div>
    </div>
  );
}
