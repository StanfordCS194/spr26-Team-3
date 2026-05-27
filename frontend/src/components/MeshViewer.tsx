import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls, Stage } from "@react-three/drei";
import { Suspense, useMemo } from "react";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import * as THREE from "three";

import { cn } from "@/lib/utils";

/** Lightweight viewer for a .ply mesh served from the backend's /data
 * static mount. Dark canvas, grid floor, OrbitControls — matches the
 * Linear/Blender feel locked in design.md.
 */
export function MeshViewer({ url, className }: { url: string; className?: string }) {
  return (
    <div
      className={cn(
        "border border-border rounded-sm overflow-hidden bg-[oklch(0.06_0_0)]",
        className,
      )}
      style={{ height: 420 }}
    >
      <Canvas camera={{ position: [4, 4, 4], fov: 45 }} dpr={[1, 2]}>
        <color attach="background" args={["#0a0a0a"]} />
        <ambientLight intensity={0.35} />
        <directionalLight position={[5, 8, 5]} intensity={0.9} />
        <Suspense fallback={null}>
          <PlyMesh url={url} />
        </Suspense>
        <Grid
          args={[20, 20]}
          cellColor="#222"
          sectionColor="#333"
          fadeDistance={30}
          fadeStrength={1.5}
          infiniteGrid
          position={[0, -0.001, 0]}
        />
        <OrbitControls makeDefault dampingFactor={0.1} enableDamping />
      </Canvas>
      <div className="absolute pointer-events-none top-2 right-2 mono text-[10px] text-muted-foreground/70 px-2 py-1 rounded-sm bg-background/60 backdrop-blur-sm">
        drag <span className="text-foreground">orbit</span> · shift+drag{" "}
        <span className="text-foreground">pan</span> · wheel{" "}
        <span className="text-foreground">zoom</span>
      </div>
    </div>
  );
}

function PlyMesh({ url }: { url: string }) {
  const geom = useMemo(() => {
    const loader = new PLYLoader();
    return new Promise<THREE.BufferGeometry>((resolve, reject) => {
      loader.load(url, (g) => {
        g.computeVertexNormals();
        g.center();
        resolve(g);
      }, undefined, reject);
    });
  }, [url]);
  const loaded = useThenable(geom);
  if (!loaded) return null;
  return (
    <mesh geometry={loaded}>
      <meshStandardMaterial color="#e0e0e0" roughness={0.55} metalness={0.05} side={THREE.DoubleSide} />
    </mesh>
  );
}

// Tiny suspense helper — react-three-fiber's Suspense expects thrown promises.
import { useState, useEffect } from "react";
function useThenable<T>(p: Promise<T>): T | null {
  const [v, setV] = useState<T | null>(null);
  useEffect(() => {
    let alive = true;
    p.then((x) => { if (alive) setV(x); }).catch(() => {});
    return () => { alive = false; };
  }, [p]);
  return v;
}
